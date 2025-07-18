package grpchandler_test

import (
	"testing"
	"time"

	grpchandler "github.com/juliog922/whatsmeow_go/src/grpc_handler"
	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/sirupsen/logrus"
	waE2E "go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
)

// === Helpers ===

func protoString(s string) *string {
	return &s
}

// === Mock Store ===

type mockStore struct {
	id *types.JID
}

func (m *mockStore) ID() *types.JID {
	return m.id
}

var _ grpchandler.WhatsAppStore = (*mockStore)(nil)

// === Mock Client ===

type mockClient struct {
	store grpchandler.WhatsAppStore
}

func (c *mockClient) GetStore() grpchandler.WhatsAppStore {
	return c.store
}

var _ grpchandler.WhatsAppClient = (*mockClient)(nil)

// === TESTS ===

func TestMakeGrpcHandler_TextMessage(t *testing.T) {
	var called bool
	var result *pb.MessageEvent

	mockBroadcast := func(msg *pb.MessageEvent) {
		called = true
		result = msg
	}

	mockHasClients := func() bool {
		return true
	}

	logger := logrus.New()

	jid := &types.JID{User: "1111", Server: "s.whatsapp.net"}
	client := &mockClient{
		store: &mockStore{id: jid},
	}

	handler := grpchandler.MakeGrpcHandler(client, nil, mockBroadcast, mockHasClients, logger)

	now := time.Now()

	msg := &events.Message{
		Info: types.MessageInfo{
			Timestamp: now,
			PushName:  "Test Name",
			MessageSource: types.MessageSource{
				Sender: types.JID{User: "2222"},
				Chat:   types.JID{User: "1111"},
			},
		},
		Message: &waE2E.Message{
			Conversation: protoString("Hello World"),
		},
	}

	handler(msg)

	if !called {
		t.Fatal("broadcastFunc was not called")
	}
	if result.Text != "Hello World" {
		t.Errorf("Expected text 'Hello World', got '%s'", result.Text)
	}
	if result.From != "2222" || result.To != "1111" {
		t.Errorf("From/To mismatch: got from=%s to=%s", result.From, result.To)
	}
	if result.Name != "Test Name" {
		t.Errorf("Expected name 'Test Name', got '%s'", result.Name)
	}
}

func TestMakeGrpcHandler_IgnoresNonMessages(t *testing.T) {
	called := false

	handler := grpchandler.MakeGrpcHandler(nil, nil,
		func(*pb.MessageEvent) {
			called = true
		},
		func() bool { return true },
		logrus.New(),
	)

	handler("not a valid event")

	if called {
		t.Fatal("Expected broadcastFunc to NOT be called")
	}
}
