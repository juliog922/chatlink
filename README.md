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
    make vpn-up / vpn-down          the Prike server tunnel (PRIKE_WG_*). It reaches
                                    only 10.20.10.100, not SAGE.
    make ssh-setup                  once: the Prike ssh key and the `prike` alias.

`make test` needs neither: SAGE is faked there.

## Test

    createdb kapa_test
    make test                     # ~6 s, no phone, no GPU

## Access to the Prike server

Once: append the PRIKE_* block to `.env`, then `make ssh-setup`. After that:

    make vpn-up        # WireGuard tunnel (asks for your local sudo password)
    ssh prike
    make vpn-down

### Prike tunnel from WSL

The same key must never be up in Windows and WSL at once. Remove the Windows
tunnel once (PowerShell as Administrator):

    & "$env:ProgramFiles\WireGuard\wireguard.exe" /uninstalltunnelservice prike
    Remove-Item -Recurse -Force "$env:ProgramData\Prike"

Then in WSL: `apt install -y wireguard-tools` and `make vpn-up`. WSL drops the
tunnel when it shuts down; `make vpn-up` again (deploy does it), or with
systemd enabled in WSL: `systemctl enable --now wg-quick@prike`.

## Deploy

    git tag v3.0.0 && git push --tags
    make deploy TAG=v3.0.0          # brings the Prike tunnel up itself

First time on the server: put a `.env` (without the PRIKE_* lines, which
only your machine needs) and `gmail_accounts.json` in `~/kapa/shared/`,
and run the first deploy with `HEALTH_WAIT_S=3600` because Tachyon downloads the models. Nightly backup: add `deploy/backup.sh` to
root's crontab. Alerts go to `ALERT_EMAILS`; set `HEARTBEAT_URL` to a
healthchecks.io check so a dead server emails you too.

## When WhatsApp breaks whatsmeow

You get a `client_outdated` or device alert. `make update-whatsmeow`,
`go build`, `make test`, a self-chat check, then tag and deploy.
