package main

import (
	"context"
	"database/sql"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/lib/pq"
	"github.com/sirupsen/logrus"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/protobuf/proto"

	pb "github.com/juliog922/meow_server/src/proto"
)

// -----------------------------------------------------------------------------
// Configuration & Helper Structs
// -----------------------------------------------------------------------------

const (
	DefaultHost = "0.0.0.0"
	DefaultPort = "50051"
)

// InMemoryFile implements io.ReadWriteSeeker for media handling in RAM.
type InMemoryFile struct {
	mu   sync.RWMutex
	data []byte
	pos  int64
}

func NewInMemoryFile() *InMemoryFile {
	return &InMemoryFile{data: make([]byte, 0, 512*1024), pos: 0}
}

func (f *InMemoryFile) Write(p []byte) (n int, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	n = len(p)
	if f.pos+int64(n) > int64(len(f.data)) {
		newSize := f.pos + int64(n)
		if newSize > int64(cap(f.data)) {
			newData := make([]byte, newSize, newSize*2)
			copy(newData, f.data)
			f.data = newData
		} else {
			f.data = f.data[:newSize]
		}
	}
	copy(f.data[f.pos:], p)
	f.pos += int64(n)
	return n, nil
}

func (f *InMemoryFile) WriteAt(p []byte, off int64) (n int, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	n = len(p)
	end := off + int64(n)
	if end > int64(len(f.data)) {
		if end > int64(cap(f.data)) {
			newData := make([]byte, end, end*2)
			copy(newData, f.data)
			f.data = newData
		} else {
			f.data = f.data[:end]
		}
	}
	copy(f.data[off:], p)
	return n, nil
}

func (f *InMemoryFile) Read(p []byte) (n int, err error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.pos >= int64(len(f.data)) {
		return 0, io.EOF
	}
	n = copy(p, f.data[f.pos:])
	f.pos += int64(n)
	return n, nil
}

func (f *InMemoryFile) ReadAt(p []byte, off int64) (n int, err error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if off >= int64(len(f.data)) {
		return 0, io.EOF
	}
	n = copy(p, f.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (f *InMemoryFile) Seek(offset int64, whence int) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = f.pos + offset
	case io.SeekEnd:
		newPos = int64(len(f.data)) + offset
	default:
		return 0, fmt.Errorf("invalid whence")
	}
	if newPos < 0 {
		return 0, fmt.Errorf("negative position")
	}
	f.pos = newPos
	return newPos, nil
}

func (f *InMemoryFile) Truncate(size int64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if size < 0 {
		return fmt.Errorf("negative size")
	}
	if size > int64(cap(f.data)) {
		newData := make([]byte, size)
		copy(newData, f.data)
		f.data = newData
	} else {
		f.data = f.data[:size]
	}
	return nil
}

func (f *InMemoryFile) Close() error { return nil }

func (f *InMemoryFile) Bytes() []byte {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.data
}

func (f *InMemoryFile) Stat() (os.FileInfo, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return &MemFileInfo{size: int64(len(f.data))}, nil
}

type MemFileInfo struct {
	size int64
}

func (m *MemFileInfo) Name() string       { return "memfile" }
func (m *MemFileInfo) Size() int64        { return m.size }
func (m *MemFileInfo) Mode() os.FileMode  { return 0644 }
func (m *MemFileInfo) ModTime() time.Time { return time.Now() }
func (m *MemFileInfo) IsDir() bool        { return false }
func (m *MemFileInfo) Sys() interface{}   { return nil }

// -----------------------------------------------------------------------------
// Server Implementation
// -----------------------------------------------------------------------------

type WhatsAppServer struct {
	pb.UnimplementedWhatsAppServiceServer
	mu        sync.RWMutex
	Container *sqlstore.Container
	DB        *sql.DB // Direct DB access for manual queries
	Clients   map[string]*whatsmeow.Client
	Listeners map[pb.WhatsAppService_StreamMessagesServer]struct{}
	Logger    *logrus.Logger
}

func NewWhatsAppServer(container *sqlstore.Container, db *sql.DB, logger *logrus.Logger) *WhatsAppServer {
	return &WhatsAppServer{
		Container: container,
		DB:        db,
		Clients:   make(map[string]*whatsmeow.Client),
		Listeners: make(map[pb.WhatsAppService_StreamMessagesServer]struct{}),
		Logger:    logger,
	}
}

// resolvePhoneFromLID queries the custom table 'whatsmeow_lid_map' to find the 'pn' (phone number).
func (s *WhatsAppServer) resolvePhoneFromLID(ctx context.Context, lidUser string) string {
	// Query based on the provided schema: lid | pn
	const query = `SELECT pn FROM whatsmeow_lid_map WHERE lid = $1 LIMIT 1`

	var phoneNumber string
	// The table stores the LID as a simple string/int (e.g. "156332699189351")
	// We pass the user part of the JID directly.
	if err := s.DB.QueryRowContext(ctx, query, lidUser).Scan(&phoneNumber); err != nil {
		if err != sql.ErrNoRows {
			s.Logger.WithError(err).Warnf("DB Error resolving LID %s", lidUser)
		}
		return ""
	}
	return phoneNumber
}

// handleWhatsAppEvent processes incoming events and resolves JIDs.
func (s *WhatsAppServer) handleWhatsAppEvent(client *whatsmeow.Client, evt interface{}) {
	msgEvt, ok := evt.(*events.Message)
	if !ok {
		return
	}

	// Ignore status updates
	if msgEvt.Info.Chat.String() == "status@broadcast" {
		return
	}

	ctx := context.Background()
	selfNumber := client.Store.ID.User

	// --- 1. RESOLVE PHONE NUMBERS (Incoming vs Outgoing) ---

	senderJID := msgEvt.Info.Sender
	chatJID := msgEvt.Info.Chat

	// Helper closure to resolve any JID to a Phone number
	getPhone := func(jid types.JID) string {
		if jid.Server == types.HiddenUserServer { // It is a LID
			phone := s.resolvePhoneFromLID(ctx, jid.User)
			if phone != "" {
				return phone
			}
			return jid.User // Fallback: return LID if not found
		}
		return jid.User // It is already a phone
	}

	var from, to string

	if msgEvt.Info.IsFromMe {
		// OUTGOING MESSAGE
		// From: Me
		// To: The person/group I sent it to (Chat JID)
		from = selfNumber
		to = getPhone(chatJID)
	} else {
		// INCOMING MESSAGE
		// From: The Sender
		// To: Me
		to = selfNumber

		// Attempt to resolve Sender
		resolvedSender := getPhone(senderJID)

		// Edge case: If 1:1 chat, sometimes Sender is LID but Chat JID is the Phone.
		// If resolution failed (still LID) AND Chat is a Phone, use Chat.
		if msgEvt.Info.IsGroup {
			from = resolvedSender
		} else {
			// 1:1 Chat
			if resolvedSender == senderJID.User && chatJID.Server == types.DefaultUserServer {
				from = chatJID.User
			} else {
				from = resolvedSender
			}
		}
	}

	// Attempt to get a display name using the RESOLVED phone number
	name := msgEvt.Info.PushName

	// We check contacts using the calculated 'from' (phone number)
	if from != "" {
		contactJID := types.NewJID(from, types.DefaultUserServer)
		if contact, err := client.Store.Contacts.GetContact(ctx, contactJID); err == nil && contact.Found {
			if contact.FullName != "" {
				name = contact.FullName
			} else if contact.PushName != "" {
				name = contact.PushName
			}
		}
	}

	// Debug Log (TODO: Remove in prod)
	s.Logger.WithFields(logrus.Fields{
		"type":        "Message",
		"final_from":  from,
		"final_to":    to,
		"original_id": msgEvt.Info.ID,
	}).Info("Event Processed")

	// --- 2. Extract Content ---
	text := ""
	if msgEvt.Message.Conversation != nil {
		text = *msgEvt.Message.Conversation
	} else if msgEvt.Message.ExtendedTextMessage != nil {
		text = *msgEvt.Message.ExtendedTextMessage.Text
	}

	pbEvent := &pb.MessageEvent{
		From:      from,
		To:        to,
		Name:      name,
		Timestamp: msgEvt.Info.Timestamp.Format(time.RFC3339),
		Text:      text,
	}

	// --- 3. Media Handling ---
	// --- 3. Media Handling ---
	img := msgEvt.Message.GetImageMessage()
	doc := msgEvt.Message.GetDocumentMessage()
	audio := msgEvt.Message.GetAudioMessage() // NUEVO

	if img != nil || doc != nil || audio != nil {
		var downloadable whatsmeow.DownloadableMessage
		var fname string

		if img != nil {
			fname = fmt.Sprintf("%s.jpg", msgEvt.Info.ID)
			downloadable = img
			pbEvent.Text = "[Image] " + img.GetCaption()
		} else if audio != nil {
			fname = fmt.Sprintf("%s.ogg", msgEvt.Info.ID)
			downloadable = audio
			pbEvent.Text = "[Audio]"
		} else {
			fname = doc.GetFileName()
			if fname == "" {
				fname = fmt.Sprintf("%s.bin", msgEvt.Info.ID)
			}
			downloadable = doc
		}

		memFile := NewInMemoryFile()
		if err := client.DownloadToFile(ctx, downloadable, memFile); err == nil {
			pbEvent.Binary = memFile.Bytes()
			pbEvent.Filename = fname
		} else {
			s.Logger.WithError(err).Error("Failed to download media")
		}
	}

	s.broadcastEvent(pbEvent)
}

func (s *WhatsAppServer) broadcastEvent(evt *pb.MessageEvent) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for stream := range s.Listeners {
		if err := stream.Send(evt); err != nil {
			s.Logger.Warn("Failed to send to stream listener")
		}
	}
}

// -----------------------------------------------------------------------------
// gRPC Methods
// -----------------------------------------------------------------------------

func (s *WhatsAppServer) StreamMessages(_ *pb.Empty, stream pb.WhatsAppService_StreamMessagesServer) error {
	s.Logger.Info("New gRPC stream listener connected")
	s.mu.Lock()
	s.Listeners[stream] = struct{}{}
	s.mu.Unlock()
	<-stream.Context().Done()
	s.mu.Lock()
	delete(s.Listeners, stream)
	s.mu.Unlock()
	s.Logger.Info("gRPC stream listener disconnected")
	return nil
}

func (s *WhatsAppServer) StartLogin(ctx context.Context, req *pb.LoginRequest) (*pb.QRCodeResponse, error) {
	s.Logger.WithField("phone", req.PhoneNumber).Info("Starting pairing code login flow")

	if req.PhoneNumber == "" {
		return &pb.QRCodeResponse{Status: "error", Code: "Phone number required"}, nil
	}

	newDev := s.Container.NewDevice()
	client := whatsmeow.NewClient(newDev, waLog.Stdout("Client", "INFO", true))

	// 1. Get the QR channel so we can wait for the connection to stabilize
	qrChan, _ := client.GetQRChannel(context.Background())

	// 2. We MUST connect before asking for a pairing code
	if err := client.Connect(); err != nil {
		s.Logger.WithError(err).Error("Connection failed")
		return &pb.QRCodeResponse{Status: "error", Code: err.Error()}, nil
	}

	// 3. Wait for the websocket to fully open (by waiting for the first QR event)
	select {
	case <-qrChan:
		s.Logger.Info("Websocket connected, requesting pairing code...")
	case <-time.After(10 * time.Second):
		s.Logger.Warn("Timeout waiting for websocket to establish")
		client.Disconnect()
		return &pb.QRCodeResponse{Status: "error", Code: "Connection timeout"}, nil
	}

	// 4. Generate the 8-character pairing code (Notice the 'ctx' argument added here)
	linkingCode, err := client.PairPhone(ctx, req.PhoneNumber, true, whatsmeow.PairClientChrome, "Chrome (Linux)")
	if err != nil {
		s.Logger.WithError(err).Error("Failed to generate pairing code")
		client.Disconnect()
		return &pb.QRCodeResponse{Status: "error", Code: err.Error()}, nil
	}

	// 5. Wait in the background for the user to type the code into their phone
	go func() {
		// As the docs state, the websocket closes after ~160 seconds if no login occurs
		timeout := time.After(3 * time.Minute)
		for {
			select {
			case <-timeout:
				s.Logger.Warn("Pairing timed out")
				client.Disconnect()
				return
			default:
				if client.IsLoggedIn() {
					s.Logger.Info("Pairing Success")
					s.registerClient(client)
					return
				}
				time.Sleep(1 * time.Second)
			}
		}
	}()

	// Return the pairing code immediately to Python
	return &pb.QRCodeResponse{Code: linkingCode, Status: "code"}, nil
}

func (s *WhatsAppServer) SendMessage(ctx context.Context, req *pb.SendRequest) (*pb.SendResponse, error) {
	client := s.getClient(req.FromJid)
	if client == nil {
		return &pb.SendResponse{Success: false, Error: "No connected client"}, nil
	}

	// JID Parsing
	var jid types.JID
	if strings.Contains(req.To, "@") {
		if parsed, err := types.ParseJID(req.To); err == nil {
			jid = parsed
		}
	}
	if jid.User == "" {
		jid = types.NewJID(req.To, types.DefaultUserServer)
	}

	var msg *waE2E.Message
	if len(req.Binary) > 0 {
		var err error
		msg, err = s.uploadMedia(ctx, client, req)
		if err != nil {
			return &pb.SendResponse{Success: false, Error: err.Error()}, nil
		}
	} else {
		msg = &waE2E.Message{Conversation: proto.String(req.Text)}
	}

	resp, err := client.SendMessage(ctx, jid, msg)
	if err != nil {
		s.Logger.WithError(err).Error("Send failed")
		return &pb.SendResponse{Success: false, Error: err.Error()}, nil
	}

	s.Logger.WithField("id", resp.ID).Info("Message sent")

	// Echo back
	s.broadcastEvent(&pb.MessageEvent{
		From: client.Store.ID.User, To: req.To, Text: req.Text, Timestamp: time.Now().Format(time.RFC3339),
	})
	return &pb.SendResponse{Success: true}, nil
}

func (s *WhatsAppServer) ListDevices(ctx context.Context, _ *pb.Empty) (*pb.DeviceList, error) {
	devs, _ := s.Container.GetAllDevices(ctx)
	var list []*pb.DeviceInfo
	for _, d := range devs {
		list = append(list, &pb.DeviceInfo{Jid: d.ID.String()})
	}
	return &pb.DeviceList{Devices: list}, nil
}

func (s *WhatsAppServer) LogoutDevice(ctx context.Context, req *pb.DeviceID) (*pb.StatusResponse, error) {
	if c := s.getClient(req.Jid); c != nil {
		c.Logout(ctx)
		return &pb.StatusResponse{Success: true}, nil
	}
	return &pb.StatusResponse{Success: false, Error: "Not found"}, nil
}

func (s *WhatsAppServer) DeleteDevice(ctx context.Context, req *pb.DeviceID) (*pb.StatusResponse, error) {
	devs, _ := s.Container.GetAllDevices(ctx)
	for _, d := range devs {
		if strings.Contains(d.ID.String(), req.Jid) {
			s.mu.Lock()
			if c, ok := s.Clients[d.ID.String()]; ok {
				c.Disconnect()
				delete(s.Clients, d.ID.String())
			}
			s.mu.Unlock()
			s.Container.DeleteDevice(ctx, d)
			return &pb.StatusResponse{Success: true}, nil
		}
	}
	return &pb.StatusResponse{Success: false}, nil
}

// -----------------------------------------------------------------------------
// Internal Logic
// -----------------------------------------------------------------------------

func (s *WhatsAppServer) registerClient(c *whatsmeow.Client) {
	s.mu.Lock()
	s.Clients[c.Store.ID.String()] = c
	s.mu.Unlock()
	c.AddEventHandler(func(evt interface{}) { s.handleWhatsAppEvent(c, evt) })
}

func (s *WhatsAppServer) getClient(jid string) *whatsmeow.Client {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if jid != "" {
		for id, c := range s.Clients {
			if strings.Contains(id, jid) {
				return c
			}
		}
	}
	for _, c := range s.Clients {
		if c.IsConnected() {
			return c
		}
	}
	return nil
}

func (s *WhatsAppServer) uploadMedia(ctx context.Context, client *whatsmeow.Client, req *pb.SendRequest) (*waE2E.Message, error) {
	mediaType := whatsmeow.MediaDocument
	ext := strings.ToLower(filepath.Ext(req.Filename))
	if ext == ".jpg" || ext == ".png" || ext == ".jpeg" {
		mediaType = whatsmeow.MediaImage
	}

	resp, err := client.Upload(ctx, req.Binary, mediaType)
	if err != nil {
		return nil, err
	}

	if mediaType == whatsmeow.MediaImage {
		return &waE2E.Message{ImageMessage: &waE2E.ImageMessage{
			URL: proto.String(resp.URL), DirectPath: proto.String(resp.DirectPath),
			MediaKey: resp.MediaKey, FileEncSHA256: resp.FileEncSHA256, FileSHA256: resp.FileSHA256,
			FileLength: proto.Uint64(uint64(len(req.Binary))),
		}}, nil
	}
	return &waE2E.Message{DocumentMessage: &waE2E.DocumentMessage{
		URL: proto.String(resp.URL), DirectPath: proto.String(resp.DirectPath),
		MediaKey: resp.MediaKey, FileEncSHA256: resp.FileEncSHA256, FileSHA256: resp.FileSHA256,
		FileLength: proto.Uint64(uint64(len(req.Binary))), FileName: proto.String(req.Filename),
	}}, nil
}

// -----------------------------------------------------------------------------
// Entrypoint
// -----------------------------------------------------------------------------

func main() {
	_ = godotenv.Load()
	logger := logrus.New()
	logger.SetFormatter(&logrus.JSONFormatter{TimestampFormat: time.RFC3339})

	sqlstore.PostgresArrayWrapper = pq.Array
	dsn := fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=disable",
		os.Getenv("POSTGRES_USER"), os.Getenv("POSTGRES_PASSWORD"), "db", "5432", os.Getenv("POSTGRES_DB"))

	// 1. Initialize DB container for Whatsmeow
	container, err := sqlstore.New(context.Background(), "postgres", dsn, nil)
	if err != nil {
		logger.Fatal(err)
	}

	// 2. Open Direct DB Connection for our custom queries (LID resolution)
	rawDB, err := sql.Open("postgres", dsn)
	if err != nil {
		logger.Fatal(err)
	}
	// Verify connection
	if err := rawDB.Ping(); err != nil {
		logger.WithError(err).Error("Failed to ping raw DB")
	}

	srv := NewWhatsAppServer(container, rawDB, logger)

	// Restore sessions
	devs, _ := container.GetAllDevices(context.Background())
	for _, d := range devs {
		c := whatsmeow.NewClient(d, waLog.Stdout("Client", "INFO", true))
		srv.registerClient(c)
		c.Connect()
	}

	lis, _ := net.Listen("tcp", "0.0.0.0:50051")
	s := grpc.NewServer()
	pb.RegisterWhatsAppServiceServer(s, srv)
	grpc_health_v1.RegisterHealthServer(s, health.NewServer())

	logger.Info("Server Started")
	if err := s.Serve(lis); err != nil {
		logger.Fatal(err)
	}
}
