#!/usr/bin/env bash
# Nightly Postgres dump (it holds the whatsmeow sessions too: a restore needs
# no re-pairing). Install in root's crontab on the server:
#   15 3 * * *  /home/julio/kapa/current/deploy/backup.sh
# Optional: BACKUP_RSYNC_TARGET=user@host:path in shared/.env copies it off the machine.
set -euo pipefail
ROOT=${KAPA_ROOT:-/home/julio/kapa}
cd "$ROOT"
mkdir -p backups
f="backups/kapa-$(date +%F).sql.gz"
docker compose --project-directory current -f current/docker-compose.yml exec -T db \
  sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' | gzip > "$f"
find backups -name 'kapa-*.sql.gz' -mtime +14 -delete
target=$(grep -E '^BACKUP_RSYNC_TARGET=' shared/.env | cut -d= -f2- || true)
[ -n "$target" ] && rsync -a "$f" "$target/"
hb=$(grep -E '^HEARTBEAT_URL=' shared/.env | cut -d= -f2- || true)
[ -s "$f" ] || { [ -n "$hb" ] && curl -fsS -m 10 "${hb%/}/fail"; exit 1; }
