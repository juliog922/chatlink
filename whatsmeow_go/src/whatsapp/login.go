package whatsapp

import (
	"context"
	"fmt"
	"time"

	grpchandler "github.com/juliog922/whatsmeow_go/src/grpc_handler"
	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/appstate"
	"go.mau.fi/whatsmeow/store/sqlstore"
	waLog "go.mau.fi/whatsmeow/util/log"
)

func (s *WhatsAppServer) handleLoginLifecycle(ctx context.Context, client *whatsmeow.Client, qrChan <-chan whatsmeow.QRChannelItem) {
	expireTimer := time.NewTimer(15 * time.Minute)
	defer expireTimer.Stop()

	for {
		select {
		case evt := <-qrChan:
			switch evt.Event {
			case "success":
				s.Logger.WithField("jid", client.Store.ID.String()).Info("Login successful")
				time.Sleep(2 * time.Second)
				*s.Clients = append(*s.Clients, client)

				wrapped := &grpchandler.ClientWrapper{Client: client}
				client.AddEventHandler(
					grpchandler.MakeGrpcHandler(wrapped, *s.Clients, s.BroadcastFunc, s.HasClientsFunc, s.Logger),
				)

				_ = client.FetchAppState(ctx, appstate.WAPatchCriticalUnblockLow, true, false)
				return

			case "timeout":
				s.Logger.Warn("QR scan timed out")
			case "error":
				s.Logger.Error("QR login failed")
			}
		case <-expireTimer.C:
			s.Logger.Warn("QR login expired (15 minutes passed)")
			client.Disconnect()
			return
		}
	}
}

func (s *WhatsAppServer) StartLogin(_ context.Context, _ *pb.Empty) (*pb.QRCodeResponse, error) {
	ctx := context.Background()

	realContainer, ok := s.Container.(*sqlstore.Container)
	if !ok {
		s.Logger.Error("Container is not a sqlstore.Container")
		return nil, fmt.Errorf("invalid container implementation")
	}

	s.Logger.Info("Creating new WhatsApp device")
	newDev := realContainer.NewDevice()

	// ✅ LOG CLIENTE
	clientLog := waLog.Stdout("Client", "INFO", true)
	client := whatsmeow.NewClient(newDev, clientLog)

	// Obtener canal QR antes de conectar
	qrChan, _ := client.GetQRChannel(ctx)

	// Registrar handler antes de conectar
	go s.handleLoginLifecycle(ctx, client, qrChan)

	// Conectar cliente
	go func() {
		if err := client.Connect(); err != nil {
			s.Logger.WithError(err).Error("Client connection failed during QR login")
		}
	}()

	// Esperar primer código QR (máximo 15s)
	select {
	case evt := <-qrChan:
		s.Logger.WithField("event", evt.Event).Info("QR channel event received")

		if evt.Event == "code" {
			s.Logger.WithField("status", "code").Info("QR code generated")

			return &pb.QRCodeResponse{
				Code:   evt.Code,
				Status: "code",
			}, nil
		}
		s.Logger.WithField("status", evt.Event).Warn("Unexpected QR event before code")
		return &pb.QRCodeResponse{Status: evt.Event}, nil

	case <-time.After(15 * time.Second):
		s.Logger.Error("Timeout waiting for first QR code")
		return &pb.QRCodeResponse{Status: "timeout"}, nil
	}
}

/*
Old Start Login
func (s *WhatsAppServer) StartLogin(_ context.Context, _ *pb.Empty) (*pb.QRCodeResponse, error) {
	ctx := context.Background()

	realContainer, ok := s.Container.(*sqlstore.Container)
	if !ok {
		s.Logger.Error("Container is not a sqlstore.Container")
		return nil, fmt.Errorf("invalid container implementation")
	}

	s.Logger.Info("Creating new WhatsApp device")
	newDev := realContainer.NewDevice()
	client := whatsmeow.NewClient(newDev, nil)

	// Obtener canal QR antes de conectar
	qrChan, _ := client.GetQRChannel(ctx)

	// Conectar cliente
	go func() {
		if err := client.Connect(); err != nil {
			s.Logger.WithError(err).Error("Client connection failed during QR login")
		}
	}()

	// Esperar QR o timeout
	timeout := time.After(15 * time.Second)

	for {
		select {
		case evt := <-qrChan:
			switch evt.Event {
			case "code":
				s.Logger.WithField("status", "code").Info("QR code generated")

				// Background handler para login resultante
				go func() {
					for e := range qrChan {
						switch e.Event {
						case "success":
							s.Logger.WithField("jid", client.Store.ID.String()).Info("Login successful")

							// Añadir a clientes activos
							*s.Clients = append(*s.Clients, client)

							// Handlers + estado
							wrapped := &grpchandler.ClientWrapper{Client: client}
							client.AddEventHandler(
								grpchandler.MakeGrpcHandler(wrapped, *s.Clients, s.BroadcastFunc, s.HasClientsFunc, s.Logger),
							)
							_ = client.FetchAppState(ctx, appstate.WAPatchCriticalUnblockLow, true, false)

						case "timeout":
							s.Logger.Warn("QR scan timed out")

						case "error":
							s.Logger.Error("QR login failed")
						}
					}
				}()

				return &pb.QRCodeResponse{
					Code:   evt.Code,
					Status: "code",
				}, nil

			case "timeout":
				s.Logger.Warn("QR scan timed out before receiving code")
				return &pb.QRCodeResponse{Status: "timeout"}, nil

			case "error":
				s.Logger.Error("QR channel returned error before code")
				return &pb.QRCodeResponse{Status: "error"}, nil
			}

		case <-timeout:
			s.Logger.Error("Timeout waiting for first QR code")
			return &pb.QRCodeResponse{Status: "timeout"}, nil
		}
	}
}
*/
