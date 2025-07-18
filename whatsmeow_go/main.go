package main

import (
	"context"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/joho/godotenv"
	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/sirupsen/logrus"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/appstate"
	"go.mau.fi/whatsmeow/store"
	"go.mau.fi/whatsmeow/store/sqlstore"
	waLog "go.mau.fi/whatsmeow/util/log"

	grpchandler "github.com/juliog922/whatsmeow_go/src/grpc_handler"
	grpcserver "github.com/juliog922/whatsmeow_go/src/grpc_server"
	"github.com/juliog922/whatsmeow_go/src/logger"
	_ "github.com/mattn/go-sqlite3"
)

var (
	container           *sqlstore.Container
	clients             []*whatsmeow.Client
	err                 error
	streamListeners     = make(map[pb.WhatsAppService_StreamMessagesServer]struct{})
	streamListenersLock = &sync.Mutex{}
)

func hasGrpcClients() bool {
	streamListenersLock.Lock()
	defer streamListenersLock.Unlock()
	return len(streamListeners) > 0
}

func main() {
	_ = godotenv.Load()
	logger := logger.InitLogger()

	host := os.Getenv("GRPC_HOST")
	if host == "" {
		host = "0.0.0.0"
	}
	port := os.Getenv("GRPC_PORT")
	if port == "" {
		port = "50051"
	}

	logger.WithFields(logrus.Fields{
		"host": host,
		"port": port,
	}).Info("Starting WhatsApp gRPC service")

	ctx := context.Background()

	dbLog := waLog.Stdout("DB", "INFO", true)
	container, err = sqlstore.New(ctx, "sqlite3", "file:examplestore.db?_foreign_keys=on", dbLog)
	if err != nil {
		logger.WithError(err).Fatal("Failed to initialize SQL store")
	}
	defer container.Close()

	// Inicializar el servidor gRPC
	waserver := grpcserver.StartGRPC(
		host,
		port,
		container,
		&clients,
		&streamListeners,
		streamListenersLock,
		logger,
		hasGrpcClients,
	)

	clientLog := waLog.Stdout("Client", "INFO", true)

	// Intentar cargar y conectar dispositivos existentes
	devices, err := container.GetAllDevices(ctx)
	if err != nil {
		logger.WithError(err).Error("Failed to load devices from store")
		devices = []*store.Device{} // evitar nil
	}

	if len(devices) == 0 {
		logger.Warn("No devices found. Service will wait for login via StartLogin.")
	} else {
		logger.WithField("count", len(devices)).Info("Devices found. Proceeding to connect.")

		for _, dev := range devices {
			c := whatsmeow.NewClient(dev, clientLog)
			clients = append(clients, c)

			c.AddEventHandler(grpchandler.MakeGrpcHandler(
				&grpchandler.ClientWrapper{Client: c},
				clients,
				waserver.BroadcastMessage,
				hasGrpcClients,
				logger,
			))

			if err := c.Connect(); err != nil {
				logger.WithFields(logrus.Fields{
					"jid": dev.ID.String(),
				}).WithError(err).Error("Failed to connect device")

				if err.Error() == "server responded with 401" ||
					err.Error() == "got 401: logged out from another device connect failure" ||
					err.Error() == "failed to send usync query: websocket not connected" {

					logger.WithField("jid", dev.ID.String()).Warn("Removing invalid session from store")
					_ = container.DeleteDevice(ctx, dev)
				}
				continue
			}

			_ = c.FetchAppState(ctx, appstate.WAPatchCriticalUnblockLow, true, false)
			logger.WithField("jid", dev.ID.String()).Info("Device connected")
		}
	}

	logger.Info("Service is ready. Waiting for messages or QR logins.")

	// Señal de apagado
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig

	logger.Warn("Interrupt received. Disconnecting clients...")
	for _, c := range clients {
		c.Disconnect()
	}
	logger.Info("Shutdown complete.")
}
