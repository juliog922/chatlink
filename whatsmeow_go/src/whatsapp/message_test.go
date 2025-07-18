package whatsapp_test

import (
	"context"
	"sync"
	"testing"
	"time"

	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/juliog922/whatsmeow_go/src/whatsapp"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"google.golang.org/grpc/metadata"
	// Reemplaza con el path real donde está tu struct WhatsAppServer
)

type mockStream struct {
	mock.Mock
	ctx context.Context
}

func (m *mockStream) Send(msg *pb.MessageEvent) error {
	args := m.Called(msg)
	return args.Error(0)
}

// grpc.ServerStream methods
func (m *mockStream) SetHeader(md metadata.MD) error  { return nil }
func (m *mockStream) SendHeader(md metadata.MD) error { return nil }
func (m *mockStream) SetTrailer(md metadata.MD)       {}
func (m *mockStream) Context() context.Context        { return m.ctx }
func (m *mockStream) SendMsg(interface{}) error       { return nil }
func (m *mockStream) RecvMsg(interface{}) error       { return nil }

func TestBroadcastMessage(t *testing.T) {
	ctx := context.Background()
	stream := &mockStream{ctx: ctx}
	msg := &pb.MessageEvent{From: "123", To: "456", Text: "Hola"}

	stream.On("Send", msg).Return(nil)

	listeners := map[pb.WhatsAppService_StreamMessagesServer]struct{}{
		stream: {},
	}

	server := &whatsapp.WhatsAppServer{
		Listeners: &listeners,
		Logger:    logrus.New(),
		Lock:      new(sync.Mutex),
	}

	server.BroadcastMessage(msg)

	// Esperamos que la goroutine corra
	time.Sleep(50 * time.Millisecond)

	stream.AssertCalled(t, "Send", msg)
}

func TestStreamMessages_AddsAndRemovesListener(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	stream := &mockStream{ctx: ctx}
	stream.On("Send", mock.Anything).Return(nil) // no se usa, pero es seguro tenerlo

	listeners := make(map[pb.WhatsAppService_StreamMessagesServer]struct{})
	lock := new(sync.Mutex)

	server := &whatsapp.WhatsAppServer{
		Listeners: &listeners,
		Logger:    logrus.New(),
		Lock:      lock,
	}

	go func() {
		_ = server.StreamMessages(&pb.Empty{}, stream)
	}()

	// Dale tiempo a la goroutine para añadir el listener
	time.Sleep(20 * time.Millisecond)

	lock.Lock()
	_, existsBefore := listeners[stream]
	lock.Unlock()
	assert.True(t, existsBefore, "stream should be added to listeners")

	// Cancelamos para que StreamMessages termine
	cancel()
	time.Sleep(20 * time.Millisecond)

	lock.Lock()
	_, existsAfter := listeners[stream]
	lock.Unlock()
	assert.False(t, existsAfter, "stream should be removed to listeners")
}
