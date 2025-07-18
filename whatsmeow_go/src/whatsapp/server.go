package whatsapp

import (
	"sync"

	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/sirupsen/logrus"
	"go.mau.fi/whatsmeow"
)

type WhatsAppServer struct {
	pb.UnimplementedWhatsAppServiceServer

	Container      DeviceStore
	Clients        *[]*whatsmeow.Client
	Listeners      *map[pb.WhatsAppService_StreamMessagesServer]struct{}
	Lock           *sync.Mutex
	Logger         *logrus.Logger
	BroadcastFunc  func(msg *pb.MessageEvent)
	HasClientsFunc func() bool
}

// Ensure it implements the gRPC interface
var _ pb.WhatsAppServiceServer = (*WhatsAppServer)(nil)
