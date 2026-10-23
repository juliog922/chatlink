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

## Test

    make test-db                  # once per boot: throwaway Postgres on localhost:55432
    make test                     # ~6 s, no phone, no GPU, no SAGE

## Deploy

    git tag v3.0.0 && git push --tags
    make deploy HOST=<ssh host> TAG=v3.0.0

First time on the server: put `.env` and `gmail_accounts.json` in
`~/kapa/shared/`, and run the first deploy with `HEALTH_WAIT_S=3600`
because Tachyon downloads the models. Nightly backup: add
`deploy/backup.sh` to root's crontab. Alerts go to `ALERT_EMAILS`; set `HEARTBEAT_URL` to a
healthchecks.io check so a dead server emails you too.

## When WhatsApp breaks whatsmeow

You get a `client_outdated` or device alert. `make update-whatsmeow`,
`go build`, `make test`, a self-chat check, then tag and deploy.
