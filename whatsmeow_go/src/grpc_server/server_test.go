package grpcserver_test

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3" // necesario para registrar el driver sqlite3

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	grpcserver "github.com/juliog922/whatsmeow_go/src/grpc_server"
	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/store/sqlstore"
	waLog "go.mau.fi/whatsmeow/util/log"
)

func TestStartGRPC_StartsServerAndReturnsWhatsAppServer(t *testing.T) {
	// Setup
	host := "127.0.0.1"
	port := "60051" // usar puerto distinto al de producción

	ctx := context.Background()
	dbLog := waLog.Noop

	dbPath := "test.db"
	defer os.Remove(dbPath) // <- borra la base al final del test

	container, err := sqlstore.New(ctx, "sqlite3", "file:"+dbPath+"?_foreign_keys=on", dbLog)
	require.NoError(t, err)
	defer container.Close()

	clients := []*whatsmeow.Client{} // se usa el tipo real aquí
	listeners := make(map[pb.WhatsAppService_StreamMessagesServer]struct{})
	lock := &sync.Mutex{}
	logger := logrus.New()

	// Stub para hasClientsFunc
	hasClients := func() bool {
		return true
	}

	// Act
	srv := grpcserver.StartGRPC(host, port, container, &clients, &listeners, lock, logger, hasClients)

	// Assert
	assert.NotNil(t, srv)
	assert.Equal(t, container, srv.Container)
	assert.Equal(t, &clients, srv.Clients)
	assert.Equal(t, &listeners, srv.Listeners)
	assert.Equal(t, lock, srv.Lock)
	assert.Equal(t, logger, srv.Logger)
	assert.NotNil(t, srv.HasClientsFunc)

	// Give some time for the server goroutine to bind
	time.Sleep(100 * time.Millisecond)
}
