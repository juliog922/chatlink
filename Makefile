# Kapa. Local: make up / test. Server: make deploy HOST=<ssh host> TAG=v3.0.0
#
# Secrets come from .env (git-ignored).

-include .env
export

KAPA_ROOT   ?= ~/kapa
TACHYON_SRC ?= ../tachyon
TACHYON_SHA  = $(shell git -C $(TACHYON_SRC) rev-parse --short HEAD)
unexport TACHYON_SHA

.PHONY: help up down logs test test-db deploy update-whatsmeow

help:
	@echo "make up                             start everything locally"
	@echo "make down / logs                    stop / follow logs"
	@echo "make test-db                        start a throwaway Postgres for the tests (Docker, port 55432)"
	@echo "make test                           scenario + unit tests (no SAGE, no GPU, no phone)"
	@echo "make deploy HOST=<ssh host> TAG=..  ship a tagged release to a server"
	@echo "make update-whatsmeow               bump whatsmeow on purpose, then test and deploy"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

test-db:
	@docker start kapa-test-db >/dev/null 2>&1 || docker run -d --name kapa-test-db \
	  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=kapa_test -p 127.0.0.1:55432:5432 postgres:16.4-alpine >/dev/null
	@until docker exec kapa-test-db pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
	@echo "Test Postgres ready on localhost:55432."

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
