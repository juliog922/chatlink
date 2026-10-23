# Kapa. Local: make up / test. Server: make deploy TAG=v3.0.0
#
# Every secret comes from .env (git-ignored): SAGE and Postgres credentials,
# the Prike WireGuard keys, the ssh key and the server's sudo password.

-include .env
export

HOST        ?= prike
KAPA_ROOT   ?= /home/julio/kapa
TACHYON_SRC ?= ../tachyon
TACHYON_SHA  = $(shell git -C $(TACHYON_SRC) rev-parse --short HEAD)
unexport TACHYON_SHA

.PHONY: help up down logs test deploy vpn-up vpn-down vpn-status ssh-setup sage-check sage-vpn-up update-whatsmeow

help:
	@echo "make up                     start everything locally (checks SAGE first)"
	@echo "make down / logs            stop / follow logs"
	@echo "make test                   scenario + unit tests (local Postgres; no SAGE, no GPU)"
	@echo "make sage-check             is SAGE reachable?    make sage-vpn-up   run SAGE_VPN_UP"
	@echo "make vpn-up / vpn-down      the Prike WireGuard tunnel, from PRIKE_WG_* in .env"
	@echo "make ssh-setup              install the Prike ssh key and the 'prike' alias, once"
	@echo "make deploy TAG=v3.0.0      ship a tagged release to the Prike server"
	@echo "make update-whatsmeow       bump whatsmeow on purpose, then test and deploy"

up: sage-check
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd chatlink_bot && python -m pytest tests -q

sage-check:
	@deploy/prike.sh sage-check || { [ -n "$$SAGE_VPN_UP" ] && $(MAKE) -s sage-vpn-up && deploy/prike.sh sage-check; }

sage-vpn-up:
	@test -n "$$SAGE_VPN_UP" || { echo "Set SAGE_VPN_UP in .env to the command that connects the SAGE VPN."; exit 1; }
	sh -c "$$SAGE_VPN_UP"

vpn-up vpn-down vpn-status ssh-setup:
	@deploy/prike.sh $@

deploy: vpn-up
	@test -n "$(TAG)" || { echo "usage: make deploy TAG=v3.0.0"; exit 1; }
	@git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null || { echo "$(TAG) is not a tag"; exit 1; }
	@test -z "$$(git -C $(TACHYON_SRC) status --porcelain)" || { echo "$(TACHYON_SRC) has uncommitted changes"; exit 1; }
	git archive --format=tar "$(TAG)" | ssh $(HOST) \
	  "mkdir -p $(KAPA_ROOT)/releases/$(TAG) && tar -x -C $(KAPA_ROOT)/releases/$(TAG)"
	git -C $(TACHYON_SRC) archive --format=tar HEAD | ssh $(HOST) \
	  "mkdir -p $(KAPA_ROOT)/tachyon/$(TACHYON_SHA) && tar -x -C $(KAPA_ROOT)/tachyon/$(TACHYON_SHA)"
	@if [ -n "$$PRIKE_SUDO_PASSWORD" ]; then \
	  printf '%s\n' "$$PRIKE_SUDO_PASSWORD" | ssh $(HOST) \
	    'read -r SUDO_PW; export SUDO_PW KAPA_ROOT=$(KAPA_ROOT); $(KAPA_ROOT)/releases/$(TAG)/deploy/deploy.sh $(TAG) $(TACHYON_SHA)'; \
	else \
	  ssh -t $(HOST) 'KAPA_ROOT=$(KAPA_ROOT) $(KAPA_ROOT)/releases/$(TAG)/deploy/deploy.sh $(TAG) $(TACHYON_SHA)'; \
	fi

update-whatsmeow:
	cd meow_server && go get go.mau.fi/whatsmeow@latest && go mod tidy
	@echo "Now: go build, make test, a self-chat check, commit, tag, deploy."
