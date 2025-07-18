package whatsapp

import (
	"context"
	"mime"
	"path/filepath"
	"strings"
	"time"

	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/sirupsen/logrus"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"
)

func (s *WhatsAppServer) StreamMessages(_ *pb.Empty, stream pb.WhatsAppService_StreamMessagesServer) error {
	s.Logger.Info("gRPC client connected to StreamMessages")

	// Register the stream listener
	s.Lock.Lock()
	(*s.Listeners)[stream] = struct{}{}
	s.Lock.Unlock()

	// Block until the client disconnects
	<-stream.Context().Done()

	// Clean up the stream listener
	s.Lock.Lock()
	delete(*s.Listeners, stream)
	s.Lock.Unlock()

	s.Logger.Info("gRPC client disconnected from StreamMessages")
	return nil
}

func (s *WhatsAppServer) BroadcastMessage(msg *pb.MessageEvent) {
	s.Lock.Lock()
	defer s.Lock.Unlock()

	for stream := range *s.Listeners {
		go func(srv pb.WhatsAppService_StreamMessagesServer) {
			if err := srv.Send(msg); err != nil {
				s.Logger.WithError(err).Warn("Failed to send message to a gRPC client")
			} else {
				s.Logger.WithFields(logrus.Fields{
					"to":   srv,
					"type": "broadcast",
				}).Debug("Message sent to gRPC client")
			}
		}(stream)
	}
}

func (s *WhatsAppServer) SendMessage(ctx context.Context, req *pb.SendRequest) (*pb.SendResponse, error) {
	if len(*s.Clients) == 0 {
		s.Logger.Warn("No connected devices available to send message")
		return &pb.SendResponse{Success: false, Error: "No connected devices"}, nil
	}

	var selectedClient *whatsmeow.Client
	if req.FromJid != "" {
		for _, c := range *s.Clients {
			fullJID := c.Store.GetJID().String()
			phone := strings.SplitN(fullJID, ":", 2)[0]
			if phone == req.FromJid {
				selectedClient = c
				break
			}
		}
		if selectedClient == nil {
			s.Logger.WithField("from_jid", req.FromJid).Warn("Requested device not found")
			return &pb.SendResponse{Success: false, Error: "Requested device not found"}, nil
		}
	} else {
		selectedClient = (*s.Clients)[0]
	}

	s.Logger.WithFields(logrus.Fields{
		"to":      req.To,
		"fromJid": req.FromJid,
		"hasFile": len(req.Binary) > 0,
	}).Info("Sending message")

	jid := types.NewJID(req.To, types.DefaultUserServer)
	var msg *waE2E.Message

	if len(req.Binary) > 0 && req.Filename != "" {
		ext := strings.ToLower(filepath.Ext(req.Filename))
		isImage := ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp"

		var mediaType whatsmeow.MediaType
		if isImage {
			mediaType = whatsmeow.MediaImage
		} else {
			mediaType = whatsmeow.MediaDocument
		}

		uploaded, err := selectedClient.Upload(ctx, req.Binary, mediaType)
		if err != nil {
			s.Logger.WithError(err).Error("Failed to upload media")
			return &pb.SendResponse{Success: false, Error: "Media upload failed: " + err.Error()}, nil
		}

		mimetype := mime.TypeByExtension(ext)
		if mimetype == "" {
			mimetype = "application/octet-stream"
		}

		if isImage {
			msg = &waE2E.Message{
				ImageMessage: &waE2E.ImageMessage{
					URL:           proto.String(uploaded.URL),
					Mimetype:      proto.String(mimetype),
					MediaKey:      uploaded.MediaKey,
					FileSHA256:    uploaded.FileSHA256,
					FileEncSHA256: uploaded.FileEncSHA256,
					DirectPath:    proto.String(uploaded.DirectPath),
					FileLength:    proto.Uint64(uint64(len(req.Binary))),
					Caption:       proto.String(req.Text),
				},
			}
		} else {
			msg = &waE2E.Message{
				DocumentMessage: &waE2E.DocumentMessage{
					URL:           proto.String(uploaded.URL),
					Mimetype:      proto.String(mimetype),
					FileName:      proto.String(req.Filename),
					FileSHA256:    uploaded.FileSHA256,
					FileLength:    proto.Uint64(uint64(len(req.Binary))),
					MediaKey:      uploaded.MediaKey,
					FileEncSHA256: uploaded.FileEncSHA256,
					DirectPath:    proto.String(uploaded.DirectPath),
				},
			}
		}
	} else {
		msg = &waE2E.Message{
			Conversation: proto.String(req.Text),
		}
	}

	_, err := selectedClient.SendMessage(ctx, jid, msg)
	if err != nil {
		s.Logger.WithError(err).WithField("jid", jid.String()).Error("Failed to send message")
		return &pb.SendResponse{Success: false, Error: err.Error()}, nil
	}

	s.Logger.WithFields(logrus.Fields{
		"to":      req.To,
		"from":    selectedClient.Store.ID.String(),
		"success": true,
	}).Info("Message sent successfully")

	// Emit message to connected clients
	s.BroadcastMessage(&pb.MessageEvent{
		From:      selectedClient.Store.ID.String(),
		To:        req.To,
		Name:      selectedClient.Store.PushName,
		Text:      req.Text,
		Timestamp: time.Now().Format("2006-01-02 15:04:05"),
		Filename:  req.Filename,
		Binary:    req.Binary,
	})

	return &pb.SendResponse{Success: true}, nil
}
