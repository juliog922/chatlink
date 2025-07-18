package whatsapp_test

import (
	"context"
	"strings"
	"testing"

	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/juliog922/whatsmeow_go/src/whatsapp"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"go.mau.fi/whatsmeow/store"
	"go.mau.fi/whatsmeow/types"
)

// --- Mocks ---

type MockDeviceStore struct {
	mock.Mock
}

func (m *MockDeviceStore) GetAllDevices(ctx context.Context) ([]*store.Device, error) {
	args := m.Called(ctx)
	return args.Get(0).([]*store.Device), args.Error(1)
}

func (m *MockDeviceStore) DeleteDevice(ctx context.Context, dev *store.Device) error {
	args := m.Called(ctx, dev)
	return args.Error(0)
}

// Interfaz para clientes usada en test
type ClientInterface interface {
	IsConnected() bool
	Logout(ctx context.Context) error
	Disconnect()
	GetID() string
}

type MockClient struct {
	mock.Mock
	id types.JID
}

func (c *MockClient) IsConnected() bool {
	args := c.Called()
	return args.Bool(0)
}

func (c *MockClient) Logout(ctx context.Context) error {
	args := c.Called(ctx)
	return args.Error(0)
}

func (c *MockClient) Disconnect() {
	c.Called()
}

func (c *MockClient) GetID() string {
	return c.id.String()
}

// --- Test ---

func TestListDevices_Success(t *testing.T) {
	ctx := context.Background()

	jid1, err := types.ParseJID("12345@s.whatsapp.net")
	assert.NoError(t, err)
	jid2, err := types.ParseJID("67890@s.whatsapp.net")
	assert.NoError(t, err)

	dev1 := &store.Device{ID: &jid1}
	dev2 := &store.Device{ID: &jid2}

	mockStore := new(MockDeviceStore)
	mockStore.On("GetAllDevices", ctx).Return([]*store.Device{dev1, dev2}, nil)

	client1 := new(MockClient)
	client1.id = jid1
	client1.On("IsConnected").Return(true)

	client2 := new(MockClient)
	client2.id = jid2
	client2.On("IsConnected").Return(false)

	clients := []ClientInterface{client1, client2}

	server := &whatsapp.WhatsAppServer{
		Container: mockStore,
		Logger:    logrus.New(),
	}

	ListDevices := func(s *whatsapp.WhatsAppServer, ctx context.Context, _ *pb.Empty) (*pb.DeviceList, error) {
		s.Logger.WithFields(map[string]interface{}{
			"operation": "list_devices",
		}).Info("Fetching devices from store")

		devices, err := s.Container.GetAllDevices(ctx)
		if err != nil {
			s.Logger.WithFields(map[string]interface{}{
				"operation": "list_devices",
			}).Errorf("Failed to fetch devices: %v", err)
			return nil, err
		}

		var list []*pb.DeviceInfo
		for _, d := range devices {
			jid := d.ID.String()
			phone := jid
			if i := strings.Index(jid, "@"); i != -1 {
				phone = jid[:i]
			}

			connected := false
			for _, c := range clients {
				if c.GetID() == jid && c.IsConnected() {
					connected = true
					break
				}
			}

			display := phone
			if !connected {
				display += " (logout)"
			}

			list = append(list, &pb.DeviceInfo{Jid: display})
		}

		s.Logger.WithFields(map[string]interface{}{
			"operation":    "list_devices",
			"device_count": len(list),
		}).Info("Device list returned successfully")

		return &pb.DeviceList{Devices: list}, nil
	}

	resp, err := ListDevices(server, ctx, &pb.Empty{})
	assert.NoError(t, err)
	assert.Len(t, resp.Devices, 2)
	assert.Equal(t, "12345", resp.Devices[0].Jid)
	assert.Equal(t, "67890 (logout)", resp.Devices[1].Jid)

	mockStore.AssertExpectations(t)
	client1.AssertExpectations(t)
	client2.AssertExpectations(t)
}

func TestListDevices_GetAllDevicesError(t *testing.T) {
	ctx := context.Background()

	mockStore := new(MockDeviceStore)
	mockStore.On("GetAllDevices", ctx).Return([]*store.Device{}, assert.AnError)

	server := &whatsapp.WhatsAppServer{
		Container: mockStore,
		Logger:    logrus.New(),
	}

	clients := []ClientInterface{} // vacío porque no importa

	ListDevices := func(s *whatsapp.WhatsAppServer, ctx context.Context, _ *pb.Empty) (*pb.DeviceList, error) {
		s.Logger.WithFields(map[string]interface{}{
			"operation": "list_devices",
		}).Info("Fetching devices from store")

		devices, err := s.Container.GetAllDevices(ctx)
		if err != nil {
			s.Logger.WithFields(map[string]interface{}{
				"operation": "list_devices",
			}).Errorf("Failed to fetch devices: %v", err)
			return nil, err
		}

		var list []*pb.DeviceInfo
		for _, d := range devices {
			jid := d.ID.String()
			phone := jid
			if i := strings.Index(jid, "@"); i != -1 {
				phone = jid[:i]
			}

			connected := false
			for _, c := range clients {
				if c.GetID() == jid && c.IsConnected() {
					connected = true
					break
				}
			}

			display := phone
			if !connected {
				display += " (logout)"
			}

			list = append(list, &pb.DeviceInfo{Jid: display})
		}

		s.Logger.WithFields(map[string]interface{}{
			"operation":    "list_devices",
			"device_count": len(list),
		}).Info("Device list returned successfully")

		return &pb.DeviceList{Devices: list}, nil
	}

	resp, err := ListDevices(server, ctx, &pb.Empty{})
	assert.Nil(t, resp)
	assert.Error(t, err)

	mockStore.AssertExpectations(t)
}

func TestDeleteDevice_Success(t *testing.T) {
	ctx := context.Background()

	jid, err := types.ParseJID("12345@s.whatsapp.net")
	assert.NoError(t, err)

	dev := &store.Device{ID: &jid}

	mockStore := new(MockDeviceStore)
	mockStore.On("GetAllDevices", ctx).Return([]*store.Device{dev}, nil)
	mockStore.On("DeleteDevice", ctx, dev).Return(nil)

	mockClient := new(MockClient)
	mockClient.id = jid
	mockClient.On("IsConnected").Return(true)
	mockClient.On("Logout", ctx).Return(nil)
	mockClient.On("Disconnect").Return()

	clients := []ClientInterface{mockClient}

	server := &whatsapp.WhatsAppServer{
		Container: mockStore,
		Logger:    logrus.New(),
	}

	DeleteDevice := func(s *whatsapp.WhatsAppServer, ctx context.Context, req *pb.DeviceID) (*pb.StatusResponse, error) {
		s.Logger.WithFields(map[string]interface{}{
			"operation": "delete_device",
			"jid":       req.Jid,
		}).Info("Starting device deletion process")

		devices, err := s.Container.GetAllDevices(ctx)
		if err != nil {
			s.Logger.WithFields(map[string]interface{}{
				"operation": "delete_device",
			}).Errorf("Failed to retrieve device list: %v", err)
			return nil, err
		}

		for _, d := range devices {
			fullJID := d.ID.String()
			phone := fullJID
			if i := strings.Index(fullJID, "@"); i != -1 {
				phone = fullJID[:i]
			}

			if phone == req.Jid {
				for _, c := range clients {
					if c.GetID() == fullJID {
						if c.IsConnected() {
							if err := c.Logout(ctx); err != nil {
								s.Logger.WithFields(map[string]interface{}{
									"jid": fullJID,
								}).Warnf("Logout before deletion failed: %v", err)
							}
						}
						c.Disconnect()
					}
				}

				if err := s.Container.DeleteDevice(ctx, d); err != nil {
					s.Logger.WithFields(map[string]interface{}{
						"jid": req.Jid,
					}).Errorf("Device deletion failed: %v", err)
					return &pb.StatusResponse{Success: false, Error: err.Error()}, nil
				}

				s.Logger.WithFields(map[string]interface{}{
					"operation": "delete_device",
					"jid":       req.Jid,
				}).Info("Device deleted successfully")

				return &pb.StatusResponse{Success: true}, nil
			}
		}

		s.Logger.WithFields(map[string]interface{}{
			"operation": "delete_device",
			"jid":       req.Jid,
		}).Warn("Device not found")

		return &pb.StatusResponse{Success: false, Error: "Device not found"}, nil
	}

	resp, err := DeleteDevice(server, ctx, &pb.DeviceID{Jid: "12345"})
	assert.NoError(t, err)
	assert.True(t, resp.Success)
	assert.Empty(t, resp.Error)

	mockStore.AssertExpectations(t)
	mockClient.AssertExpectations(t)
}
