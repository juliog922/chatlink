# Kapa

Covers a busy salesman's WhatsApp and Gmail chats and turns them into orders
he only has to validate. What it does, scenario by scenario: [SPEC.md](SPEC.md).

    meow_server/    Go: WhatsApp through whatsmeow, gRPC to the bot
    chatlink_bot/   Python: ingest, the loop, the turn, delivery, alerts
    deploy/         Tachyon image, deploy.sh, backup.sh

## Run locally

Needs Docker with the NVIDIA runtime and a GPU of compute capability 8.0 or
newer (RTX 30-series and later), plus a Tachyon checkout next to this repo.

    cp .env.example .env          # fill it; on a small GPU: KAPA_CHAT_MODEL=gemma-4-e2b
    make up                       # first start downloads the models
    curl localhost:8000/api/healthz   # the admin API, localhost only

Live check without a client: link your number (`login` in self-chat), then
write to yourself; Kapa treats you as the client.

## VPNs

Two different networks, both configured in `.env`:

    make sage-check / sage-vpn-up   SAGE (SQL Server). make up checks it first and
                                    runs SAGE_VPN_UP when SAGE does not answer.
    make prike-check                does the Prike server answer?

The Prike tunnel runs on Windows, installed by Asterio's `conectar-prike.ps1`
(PowerShell as Administrator, see his LEEME). WSL reaches the server through it. It cannot run inside WSL: the server
answers WireGuard over IPv6 only, and WSL's NAT network has no IPv6.
Once: `make ssh-setup` for the `ssh prike` alias.

`make test` needs neither: SAGE is faked there.

## Test

    createdb kapa_test
    make test                     # ~6 s, no phone, no GPU

## Deploy

    git tag v3.0.0 && git push --tags
    make deploy TAG=v3.0.0          # checks the Prike tunnel first

First time on the server: put a `.env` (without the PRIKE_* lines, which
only your machine needs) and `gmail_accounts.json` in `~/kapa/shared/`,
and run the first deploy with `HEALTH_WAIT_S=3600` because Tachyon downloads the models. Nightly backup: add `deploy/backup.sh` to
root's crontab. Alerts go to `ALERT_EMAILS`; set `HEARTBEAT_URL` to a
healthchecks.io check so a dead server emails you too.

## When WhatsApp breaks whatsmeow

You get a `client_outdated` or device alert. `make update-whatsmeow`,
`go build`, `make test`, a self-chat check, then tag and deploy.
