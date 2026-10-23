#!/usr/bin/env bash
# Runs ON THE SERVER, called by `make deploy` after it copied the release.
#   deploy.sh TAG TACHYON_SHA
# Layout under $KAPA_ROOT (default ~/kapa):
#   shared/.env, shared/gmail_accounts.json   written once by hand, never in git
#   releases/<TAG>/                           one directory per release
#   tachyon/<sha>/                            the Tachyon source of that release
#   current -> releases/<TAG>                 what is running
#   backups/                                  pg_dump before every deploy, and nightly
# Brings the new release up, waits until it is healthy, else brings the
# previous one back and pings the heartbeat's /fail.
set -euo pipefail
TAG=$1; TSHA=$2
ROOT=${KAPA_ROOT:-$HOME/kapa}
# Docker through sudo by default (make deploy uses ssh -t, so sudo can ask).
dk() { ${DOCKER:-sudo docker} "$@"; }
WAIT_S=${HEALTH_WAIT_S:-300}      # first deploy: HEALTH_WAIT_S=3600 (models download)
NEW=$ROOT/releases/$TAG
cd "$ROOT"

[ -f shared/.env ] || { echo "Missing $ROOT/shared/.env (copy .env.example and fill it)."; exit 1; }
command -v nvidia-smi >/dev/null || { echo "No NVIDIA driver on this host: Tachyon needs one."; exit 1; }
[ -f shared/gmail_accounts.json ] || echo '{}' > shared/gmail_accounts.json
ln -sfn "$ROOT/shared/.env" "$NEW/.env"
ln -sfn "$ROOT/shared/gmail_accounts.json" "$NEW/chatlink_bot/gmail_accounts.json"
export TACHYON_SRC=$ROOT/tachyon/$TSHA
mkdir -p backups

compose() { local dir=$1; shift; dk compose --project-directory "$dir" -f "$dir/docker-compose.yml" "$@"; }
env_get() { grep -E "^$1=" shared/.env | tail -1 | cut -d= -f2-; }

PREV=$(readlink current || true)
if [ -n "$PREV" ] && [ "$PREV" != "$NEW" ]; then
  echo "==> Backing up Postgres"
  compose "$PREV" exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' \
    | gzip > "backups/predeploy-$(date +%F-%H%M)-$TAG.sql.gz"
fi

echo "==> Building and starting $TAG"
compose "$NEW" build
compose "$NEW" up -d --remove-orphans

echo "==> Waiting up to ${WAIT_S}s for the bot to be healthy"
bot=$(compose "$NEW" ps -q chatlink_bot)
for _ in $(seq 1 $((WAIT_S / 5))); do
  status=$(dk inspect --format '{{.State.Health.Status}}' "$bot" 2>/dev/null || echo starting)
  if [ "$status" = healthy ]; then
    ln -sfn "$NEW" current
    echo "==> $TAG is live."
    exit 0
  fi
  sleep 5
done

echo "==> $TAG did not become healthy."
compose "$NEW" logs --tail 80 chatlink_bot tachyon || true
if [ -n "$PREV" ] && [ "$PREV" != "$NEW" ]; then
  echo "==> Rolling back to $(basename "$PREV")"
  compose "$PREV" up -d --remove-orphans
fi
hb=$(env_get HEARTBEAT_URL || true)
[ -n "$hb" ] && curl -fsS -m 10 "${hb%/}/fail" >/dev/null || true
exit 1
