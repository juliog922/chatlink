package grpcserver

import (
	"fmt"
	"net"
	"sync"

	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/juliog922/whatsmeow_go/src/whatsapp"
	"github.com/sirupsen/logrus"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"google.golang.org/grpc"
)

func StartGRPC(
	host, port string,
	container *sqlstore.Container,
	clients *[]*whatsmeow.Client,
	listeners *map[pb.WhatsAppService_StreamMessagesServer]struct{},
	lock *sync.Mutex,
	logger *logrus.Logger,
	hasClientsFunc func() bool,
) *whatsapp.WhatsAppServer {
	addr := fmt.Sprintf("%s:%s", host, port)

	listener, err := net.Listen("tcp", addr)
	if err != nil {
		logger.WithFields(logrus.Fields{
			"address": addr,
		}).WithError(err).Fatal("Failed to start gRPC listener")
	}

	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(64*1024*1024),
		grpc.MaxSendMsgSize(64*1024*1024),
	)

	srv := &whatsapp.WhatsAppServer{
		Container:      container,
		Clients:        clients,
		Listeners:      listeners,
		Lock:           lock,
		Logger:         logger,
		HasClientsFunc: hasClientsFunc,
		BroadcastFunc: func(msg *pb.MessageEvent) {
			lock.Lock()
			defer lock.Unlock()

			for stream := range *listeners {
				err := stream.Send(msg)
				if err != nil {
					logger.WithError(err).Warn("Failed to send message to stream client, removing...")
					delete(*listeners, stream)
				}
			}
		},
	}

	pb.RegisterWhatsAppServiceServer(server, srv)

	logger.WithField("address", addr).Info("gRPC server started")

	go func() {
		if err := server.Serve(listener); err != nil {
			logger.WithError(err).Fatal("gRPC server failed")
		}
	}()

	return srv
}
