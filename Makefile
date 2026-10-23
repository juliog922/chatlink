# Kapa. Local: make up / test. Server: make deploy HOST=<ssh host> TAG=v3.0.0
#
# Secrets come from .env (git-ignored).

-include .env
export

KAPA_ROOT        ?= ~/kapa
TACHYON_UPSTREAM ?= ~/projects/tachyon

.PHONY: help up down logs test test-models test-db deploy vendor-tachyon update-whatsmeow

help:
	@echo "make up                             start everything locally"
	@echo "make down / logs                    stop / follow logs"
	@echo "make test-db                        start a throwaway Postgres for the tests (Docker, port 55432)"
	@echo "make test                           scenario + unit tests, models faked (seconds)"
	@echo "make test-models                    the same scenarios with real Gemma + Laya (needs tachyon serve)"
	@echo "make deploy HOST=<ssh host> TAG=..  ship a tagged release to a server"
	@echo "make vendor-tachyon                 refresh ./tachyon from ~/projects/tachyon, then commit"
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

test-models:
	cd chatlink_bot && python -m pytest tests -q -m models -rs

deploy:
	@test -n "$(HOST)" && test -n "$(TAG)" || { echo "usage: make deploy HOST=<ssh host> TAG=v3.0.0"; exit 1; }
	@git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null || { echo "$(TAG) is not a tag"; exit 1; }
	git archive --format=tar "$(TAG)" | ssh $(HOST) \
	  "mkdir -p $(KAPA_ROOT)/releases/$(TAG) && tar -x -C $(KAPA_ROOT)/releases/$(TAG)"
	ssh -t $(HOST) 'KAPA_ROOT=$(KAPA_ROOT) $(KAPA_ROOT)/releases/$(TAG)/deploy/deploy.sh $(TAG)'

# Tachyon lives in its own repo; ./tachyon is a build copy (source only, no
# test fixtures) so Kapa builds and deploys from one tree.
vendor-tachyon:
	@test -z "$$(git -C $(TACHYON_UPSTREAM) status --porcelain)" || { echo "$(TACHYON_UPSTREAM) has uncommitted changes"; exit 1; }
	rm -rf tachyon && mkdir tachyon
	git -C $(TACHYON_UPSTREAM) archive --format=tar HEAD -- . ':(exclude,glob)crates/*/tests/fixtures/**' ':(exclude).github' | tar -x -C tachyon
	@printf 'Copied from %s at commit %s (%s) by make vendor-tachyon.\nDo not edit here: change Tachyon in its own repo, then run make vendor-tachyon.\n' \
	  "$(TACHYON_UPSTREAM)" "$$(git -C $(TACHYON_UPSTREAM) rev-parse --short HEAD)" "$$(git -C $(TACHYON_UPSTREAM) log -1 --format=%cs)" > tachyon/VENDORED
	@cat tachyon/VENDORED; echo "Now: git add tachyon && git commit"

update-whatsmeow:
	cd meow_server && go get go.mau.fi/whatsmeow@latest && go mod tidy
	@echo "Now: go build, make test, a self-chat check, commit, tag, deploy."
