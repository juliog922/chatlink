# Makefile for ChatLink Project

# Automatically detect the directory name to use as the project name (e.g., chatlink)
# This is used to target the specific volumes docker-compose creates (e.g., chatlink_postgres_data)
PROJECT_NAME := $(shell basename $(CURDIR))

.PHONY: help up up-full down logs clean-semi clean-all update-whatsmeow

# Default target
help:
	@echo "ChatLink Management Commands:"
	@echo "  make up               - Start all services (detached, builds if changed)"
	@echo "  make up-full          - Start all services (detached, builds if changed) and update 'whatsmeow' dependency in meow_server"
	@echo "  make down             - Stop all services"
	@echo "  make logs             - Follow logs for all services"
	@echo "  make clean-semi       - Stop services & remove ONLY Postgres/Qdrant volumes (Preserves AI models)"
	@echo "  make clean-all        - Stop services & remove EVERYTHING (including large AI models)"
	@echo "  make update-whatsmeow - Update 'whatsmeow' dependency in meow_server"

up:
	@echo "Starting services..."
	docker compose up -d --build

up-full:
	@echo "Updating whatsmeow dependency in ./meow_server..."
	cd meow_server && go get go.mau.fi/whatsmeow@latest && go mod tidy
	@echo "✅ Dependency updated in go.mod."
	@echo "Starting services..."
	docker compose up -d --build

down:
	@echo "Stopping services..."
	docker compose down

logs:
	docker compose logs -f

clean-semi:
	@echo "Stopping services..."
	@echo "Removing Postgres and Qdrant volumes..."
	docker compose down db qdrant -v
	@echo "Removing Postgres and Qdrant volumes..."
	@echo "Done."

clean-all:
	@echo "WARNING: This will remove ALL data, including downloaded AI models."
	@echo "Stopping services and removing all volumes..."
	docker compose down --volumes --remove-orphans
	@echo "All cleaned."

update-whatsmeow:
	@echo "Updating whatsmeow dependency in ./meow_server..."
	cd meow_server && go get go.mau.fi/whatsmeow@latest && go mod tidy
	@echo "✅ Dependency updated in go.mod."
	@echo "ℹ️  Run 'make up' to rebuild the container with the new version."