package logger

import (
	"os"
	"time"

	"github.com/sirupsen/logrus"
)

func InitLogger() *logrus.Logger {
	var logger = logrus.New()

	logger.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: time.RFC3339,
	})
	levelStr := os.Getenv("LOG_LEVEL")
	if levelStr == "" {
		levelStr = "info"
	}
	level, err := logrus.ParseLevel(levelStr)
	if err != nil {
		logger.Warnf("Invalid LOG_LEVEL '%s', defaulting to INFO", levelStr)
		level = logrus.InfoLevel
	}
	logger.SetLevel(level)

	return logger
}
