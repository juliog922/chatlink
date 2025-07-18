package whatsapp

import (
	"context"
	"fmt"
	"strings"

	pb "github.com/juliog922/whatsmeow_go/src/proto"
	"github.com/sirupsen/logrus"
	"go.mau.fi/whatsmeow/store"
)

type DeviceStore interface {
	GetAllDevices(ctx context.Context) ([]*store.Device, error)
	DeleteDevice(ctx context.Context, dev *store.Device) error
}

func (s *WhatsAppServer) ListDevices(ctx context.Context, _ *pb.Empty) (*pb.DeviceList, error) {
	s.Logger.WithFields(logrus.Fields{
		"operation": "list_devices",
	}).Info("Fetching devices from store")

	devices, err := s.Container.GetAllDevices(ctx)
	if err != nil {
		s.Logger.WithFields(logrus.Fields{
			"operation": "list_devices",
		}).WithError(err).Error("Failed to fetch devices")
		return nil, fmt.Errorf("failed to get devices: %w", err)
	}

	var list []*pb.DeviceInfo
	for _, d := range devices {
		jid := d.ID.String()
		phone := strings.SplitN(jid, ":", 2)[0]

		connected := false
		for _, c := range *s.Clients {
			if c.Store.ID.String() == jid && c.IsConnected() {
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

	s.Logger.WithFields(logrus.Fields{
		"operation":    "list_devices",
		"device_count": len(list),
	}).Info("Device list returned successfully")

	return &pb.DeviceList{Devices: list}, nil
}

func (s *WhatsAppServer) DeleteDevice(ctx context.Context, req *pb.DeviceID) (*pb.StatusResponse, error) {
	s.Logger.WithFields(logrus.Fields{
		"operation": "delete_device",
		"jid":       req.Jid,
	}).Info("Starting device deletion process")

	devices, err := s.Container.GetAllDevices(ctx)
	if err != nil {
		s.Logger.WithFields(logrus.Fields{
			"operation": "delete_device",
		}).WithError(err).Error("Failed to retrieve device list")
		return nil, fmt.Errorf("failed to fetch devices: %w", err)
	}

	for _, d := range devices {
		fullJID := d.ID.String()
		phone := strings.SplitN(fullJID, ":", 2)[0]

		if phone == req.Jid {
			for _, c := range *s.Clients {
				if c.Store.ID.String() == fullJID {
					if c.IsConnected() {
						if err := c.Logout(ctx); err != nil {
							s.Logger.WithFields(logrus.Fields{
								"jid": fullJID,
							}).WithError(err).Warn("Logout before deletion failed")
						}
					}
					c.Disconnect()
				}
			}

			if err := s.Container.DeleteDevice(ctx, d); err != nil {
				s.Logger.WithFields(logrus.Fields{
					"jid": req.Jid,
				}).WithError(err).Error("Device deletion failed")
				return &pb.StatusResponse{Success: false, Error: err.Error()}, nil
			}

			s.Logger.WithFields(logrus.Fields{
				"operation": "delete_device",
				"jid":       req.Jid,
			}).Info("Device deleted successfully")

			return &pb.StatusResponse{Success: true}, nil
		}
	}

	s.Logger.WithFields(logrus.Fields{
		"operation": "delete_device",
		"jid":       req.Jid,
	}).Warn("Device not found")

	return &pb.StatusResponse{Success: false, Error: "Device not found"}, nil
}
