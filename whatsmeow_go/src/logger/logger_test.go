package logger_test

import (
	"os"
	"testing"

	"github.com/juliog922/whatsmeow_go/src/logger"
	"github.com/sirupsen/logrus"
)

func TestInitLogger_DefaultLevel(t *testing.T) {
	_ = os.Unsetenv("LOG_LEVEL")

	log := logger.InitLogger()
	if log.GetLevel() != logrus.InfoLevel {
		t.Errorf("Expected default level to be INFO, got %s", log.GetLevel())
	}
}

func TestInitLogger_DebugLevel(t *testing.T) {
	_ = os.Setenv("LOG_LEVEL", "debug")

	log := logger.InitLogger()
	if log.GetLevel() != logrus.DebugLevel {
		t.Errorf("Expected level to be DEBUG, got %s", log.GetLevel())
	}
}

func TestInitLogger_InvalidLevel(t *testing.T) {
	_ = os.Setenv("LOG_LEVEL", "invalid_level")

	log := logger.InitLogger()
	if log.GetLevel() != logrus.InfoLevel {
		t.Errorf("Expected fallback level to be INFO, got %s", log.GetLevel())
	}
}
