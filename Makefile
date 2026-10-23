# Kapa. Local: make up / test. Server: make deploy HOST=<ssh host> TAG=v3.0.0
#
# Secrets come from .env (git-ignored).

-include .env
export

KAPA_ROOT   ?= ~/kapa
TACHYON_SRC ?= ../tachyon
TACHYON_SHA  = $(shell git -C $(TACHYON_SRC) rev-parse --short HEAD)
unexport TACHYON_SHA

.PHONY: help up down logs test deploy update-whatsmeow

help:
	@echo "make up                             start everything locally"
	@echo "make down / logs                    stop / follow logs"
	@echo "make test                           scenario + unit tests (local Postgres; no SAGE, no GPU)"
	@echo "make deploy HOST=<ssh host> TAG=..  ship a tagged release to a server"
	@echo "make update-whatsmeow               bump whatsmeow on purpose, then test and deploy"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd chatlink_bot && python -m pytest tests -q

deploy:
	@test -n "$(HOST)" && test -n "$(TAG)" || { echo "usage: make deploy HOST=<ssh host> TAG=v3.0.0"; exit 1; }
	@git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null || { echo "$(TAG) is not a tag"; exit 1; }
	@test -z "$$(git -C $(TACHYON_SRC) status --porcelain)" || { echo "$(TACHYON_SRC) has uncommitted changes"; exit 1; }
	git archive --format=tar "$(TAG)" | ssh $(HOST) \
	  "mkdir -p $(KAPA_ROOT)/releases/$(TAG) && tar -x -C $(KAPA_ROOT)/releases/$(TAG)"
	git -C $(TACHYON_SRC) archive --format=tar HEAD | ssh $(HOST) \
	  "mkdir -p $(KAPA_ROOT)/tachyon/$(TACHYON_SHA) && tar -x -C $(KAPA_ROOT)/tachyon/$(TACHYON_SHA)"
	ssh -t $(HOST) 'KAPA_ROOT=$(KAPA_ROOT) $(KAPA_ROOT)/releases/$(TAG)/deploy/deploy.sh $(TAG) $(TACHYON_SHA)'

update-whatsmeow:
	cd meow_server && go get go.mau.fi/whatsmeow@latest && go mod tidy
	@echo "Now: go build, make test, a self-chat check, commit, tag, deploy."
