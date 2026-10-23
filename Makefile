# Kapa. Local: make up / test. Server: make deploy HOST=prike TAG=v3.0.0
#
# HOST is an ssh alias (conectar-prike.sh creates `prike`; the WireGuard
# tunnel must be up). The server never builds from git: this machine sends
# the tagged tree and the Tachyon source over ssh.

HOST        ?= prike
KAPA_ROOT   ?= /home/julio/kapa
TACHYON_SRC ?= ../tachyon
TACHYON_SHA  = $(shell git -C $(TACHYON_SRC) rev-parse --short HEAD)

.PHONY: help up down logs test deploy update-whatsmeow

help:
	@echo "make up                         start everything locally (needs an NVIDIA GPU)"
	@echo "make down / logs                stop / follow logs"
	@echo "make test                       scenario + unit tests (needs a local Postgres)"
	@echo "make deploy HOST=prike TAG=v..  ship a tagged release to the server"
	@echo "make update-whatsmeow           bump whatsmeow on purpose, then test and deploy"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd chatlink_bot && python -m pytest tests -q

deploy:
	@test -n "$(TAG)" || { echo "usage: make deploy HOST=prike TAG=v3.0.0"; exit 1; }
	@git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null || { echo "$(TAG) is not a tag"; exit 1; }
	@test -z "$$(git -C $(TACHYON_SRC) status --porcelain)" || { echo "$(TACHYON_SRC) has uncommitted changes"; exit 1; }
	git archive --format=tar "$(TAG)" | ssh $(HOST) \
	  "mkdir -p $(KAPA_ROOT)/releases/$(TAG) && tar -x -C $(KAPA_ROOT)/releases/$(TAG)"
	git -C $(TACHYON_SRC) archive --format=tar HEAD | ssh $(HOST) \
	  "mkdir -p $(KAPA_ROOT)/tachyon/$(TACHYON_SHA) && tar -x -C $(KAPA_ROOT)/tachyon/$(TACHYON_SHA)"
	ssh -t $(HOST) "KAPA_ROOT=$(KAPA_ROOT) $(KAPA_ROOT)/releases/$(TAG)/deploy/deploy.sh $(TAG) $(TACHYON_SHA)"

update-whatsmeow:
	cd meow_server && go get go.mau.fi/whatsmeow@latest && go mod tidy
	@echo "Now: go build, make test, a self-chat check, commit, tag, deploy."
