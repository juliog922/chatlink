package grpchandler

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/sirupsen/logrus"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	"google.golang.org/protobuf/proto"
)

type DeviceWrapper struct {
	JID *types.JID
}

func (d *DeviceWrapper) ID() *types.JID {
	return d.JID
}

type ClientWrapper struct {
	Client *whatsmeow.Client
}

func (cw *ClientWrapper) GetStore() WhatsAppStore {
	return &DeviceWrapper{JID: cw.Client.Store.ID}
}

type WhatsAppStore interface {
	ID() *types.JID
}

type WhatsAppClient interface {
	GetStore() WhatsAppStore
}

type inMemoryFile struct {
	buf *bytes.Buffer
	pos int64
}

func newInMemoryFile() *inMemoryFile {
	return &inMemoryFile{
		buf: bytes.NewBuffer(make([]byte, 0, 512*1024)), // buffer inicial
		pos: 0,
	}
}

func (f *inMemoryFile) Write(p []byte) (int, error) {
	return f.buf.Write(p)
}

func (f *inMemoryFile) WriteAt(p []byte, off int64) (int, error) {
	end := int(off) + len(p)
	current := f.buf.Bytes()

	if len(current) < end {
		newBuf := make([]byte, end)
		copy(newBuf, current)
		copy(newBuf[off:], p)
		f.buf = bytes.NewBuffer(newBuf)
	} else {
		tmp := make([]byte, len(current))
		copy(tmp, current)
		copy(tmp[off:], p)
		f.buf = bytes.NewBuffer(tmp)
	}

	return len(p), nil
}

func (f *inMemoryFile) Read(p []byte) (int, error) {
	r := bytes.NewReader(f.buf.Bytes())
	r.Seek(f.pos, io.SeekStart)
	n, err := r.Read(p)
	f.pos += int64(n)
	return n, err
}

func (f *inMemoryFile) ReadAt(p []byte, off int64) (int, error) {
	return bytes.NewReader(f.buf.Bytes()).ReadAt(p, off)
}

func (f *inMemoryFile) Seek(offset int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case io.SeekStart:
		abs = offset
	case io.SeekCurrent:
		abs = f.pos + offset
	case io.SeekEnd:
		abs = int64(f.buf.Len()) + offset
	default:
		return 0, fmt.Errorf("invalid whence")
	}
	if abs < 0 {
		return 0, fmt.Errorf("negative position")
	}
	f.pos = abs
	return abs, nil
}

func (f *inMemoryFile) Truncate(size int64) error {
	if size < 0 {
		return fmt.Errorf("invalid size")
	}
	data := f.buf.Bytes()
	if int64(len(data)) > size {
		f.buf = bytes.NewBuffer(data[:size])
	} else if int64(len(data)) < size {
		padding := make([]byte, size-int64(len(data)))
		f.buf.Write(padding)
	}
	return nil
}

func (f *inMemoryFile) Close() error {
	return nil
}

func (f *inMemoryFile) Stat() (os.FileInfo, error) {
	return dummyStat{size: f.buf.Len()}, nil
}

type dummyStat struct {
	size int
}

func (d dummyStat) Name() string       { return "memoryfile" }
func (d dummyStat) Size() int64        { return int64(d.size) }
func (d dummyStat) Mode() os.FileMode  { return 0444 }
func (d dummyStat) ModTime() time.Time { return time.Now() }
func (d dummyStat) IsDir() bool        { return false }
func (d dummyStat) Sys() interface{}   { return nil }

// makeGrpcHandler returns a message event handler function for a given WhatsApp client.
// It handles incoming messages (text or media), performs basic parsing, downloads media, and broadcasts the event.
func MakeGrpcHandler(
	client WhatsAppClient,
	clients []*whatsmeow.Client,
	broadcastFunc func(msg *pb.MessageEvent),
	hasClientsFunc func() bool,
	logger *logrus.Logger,
) func(evt interface{}) {
	return func(evt interface{}) {
		msgEvt, ok := evt.(*events.Message)
		if !ok {
			logger.Debug("Received non-message event, ignoring")
			return
		}

		if client == nil || client.GetStore() == nil || client.GetStore().ID() == nil {
			logger.Warn("Client or Store not initialized properly, message ignored")
			return
		}

		selfNumber := client.GetStore().ID().User
		sender := msgEvt.Info.Sender.User
		chat := msgEvt.Info.Chat.User

		var from, to string
		if sender == selfNumber {
			from = sender
			to = chat
		} else {
			from = sender
			to = selfNumber
		}

		timestamp := msgEvt.Info.Timestamp.Format("2006-01-02_150405")
		name := msgEvt.Info.PushName

		// Extract text content if available
		var text string
		switch {
		case msgEvt.Message.GetConversation() != "":
			text = msgEvt.Message.GetConversation()
		case msgEvt.Message.GetExtendedTextMessage() != nil:
			text = msgEvt.Message.GetExtendedTextMessage().GetText()
		case msgEvt.Message.GetButtonsResponseMessage() != nil:
			text = msgEvt.Message.GetButtonsResponseMessage().GetSelectedDisplayText()
		case msgEvt.Message.GetTemplateButtonReplyMessage() != nil:
			text = msgEvt.Message.GetTemplateButtonReplyMessage().GetSelectedDisplayText()
		}

		if text != "" {
			logger.WithFields(logrus.Fields{
				"from": from,
				"to":   to,
				"text": text,
			}).Info("Text message received")

			if !hasClientsFunc() {
				logger.Warn("No connected gRPC clients, skipping broadcast")
				return
			}

			broadcastFunc(&pb.MessageEvent{
				From:      from,
				To:        to,
				Name:      name,
				Text:      text,
				Timestamp: timestamp,
			})
			return
		}

		// Handle media messages
		logger.WithFields(logrus.Fields{
			"from": from,
			"to":   to,
		}).Info("Media message received")

		_ = os.MkdirAll("/tmp/media", 0755)
		base := fmt.Sprintf("/tmp/media/%s_%s", from, timestamp)

		type media struct {
			msg      proto.Message
			filename string
			ext      string
		}

		var m *media

		switch {
		case msgEvt.Message.GetImageMessage() != nil:
			m = &media{msgEvt.Message.GetImageMessage(), base + ".jpg", ".jpg"}
		case msgEvt.Message.GetVideoMessage() != nil:
			m = &media{msgEvt.Message.GetVideoMessage(), base + ".mp4", ".mp4"}
		case msgEvt.Message.GetDocumentMessage() != nil:
			doc := msgEvt.Message.GetDocumentMessage()
			ext := ".bin"
			if fn := doc.GetFileName(); strings.Contains(fn, ".") {
				ext = fn[strings.LastIndex(fn, "."):]
			}
			m = &media{doc, base + ext, ext}
		case msgEvt.Message.GetAudioMessage() != nil:
			m = &media{msgEvt.Message.GetAudioMessage(), base + ".ogg", ".ogg"}
		default:
			logger.Warn("Unrecognized media type, message skipped")
			return
		}

		if len(clients) == 0 {
			logger.Warn("No WhatsApp clients connected, skipping download")
			return
		}

		ctx := context.Background()

		dm, ok := m.msg.(whatsmeow.DownloadableMessage)
		if !ok {
			logger.Error("Message is not a downloadable type")
			return
		}

		// Try to get expected file size
		var expectedSize int64 = -1
		switch msg := m.msg.(type) {
		case *waE2E.ImageMessage:
			expectedSize = int64(msg.GetFileLength())
		case *waE2E.VideoMessage:
			expectedSize = int64(msg.GetFileLength())
		case *waE2E.AudioMessage:
			expectedSize = int64(msg.GetFileLength())
		case *waE2E.DocumentMessage:
			expectedSize = int64(msg.GetFileLength())
		}

		logger.WithFields(logrus.Fields{
			"file":          m.filename,
			"expectedBytes": expectedSize,
		}).Info("Downloading media")

		f := newInMemoryFile()
		if err := clients[0].DownloadToFile(ctx, dm, f); err != nil {
			logger.WithError(err).Error("Failed to download media file")
			return
		}
		data := f.buf.Bytes()

		logger.WithFields(logrus.Fields{
			"file":     m.filename,
			"bytes":    len(data),
			"from":     from,
			"to":       to,
			"filename": filepath.Base(m.filename),
		}).Info("Media file downloaded successfully")

		broadcastFunc(&pb.MessageEvent{
			From:      from,
			To:        to,
			Name:      name,
			Timestamp: timestamp,
			Text:      "MEDIA:" + filepath.Base(m.filename),
			Binary:    data,
			Filename:  filepath.Base(m.filename),
		})
	}
}
