#!/bin/sh
# Pull Kapa's models once, start the daemon, keep them pinned in VRAM.
set -eu
# The bot runs this same binary as a client of the daemon.
cp /usr/local/bin/tachyon /run/tachyon/tachyon
for m in "$KAPA_CHAT_MODEL" "$KAPA_LAYA_MODEL"; do
  [ -f "$TACHYON_HOME/$m/weights.wpk" ] || tachyon pull "$m"
done
rm -f "$TACHYON_SOCKET"
tachyon serve --socket="$TACHYON_SOCKET" &
pid=$!
i=0
until [ -S "$TACHYON_SOCKET" ]; do
  i=$((i + 1)); [ "$i" -gt 60 ] && { echo "tachyon serve did not start"; exit 1; }
  sleep 1
done
tachyon load "$KAPA_CHAT_MODEL" --keep-alive=never --pin
tachyon load "$KAPA_LAYA_MODEL" --keep-alive=never --pin
wait "$pid"
