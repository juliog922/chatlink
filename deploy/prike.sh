#!/usr/bin/env bash
# The way to the Prike server, built from the PRIKE_* variables in .env
# (git-ignored), so no key file lives anywhere else. Called by the Makefile:
#
#   prike.sh vpn-up       write the WireGuard config and bring the tunnel up
#   prike.sh vpn-down     take it down
#   prike.sh vpn-status   exit 0 if the server answers through the tunnel
#   prike.sh ssh-setup    install the ssh key, the server's host key and the `prike` alias
#   prike.sh sage-check   exit 0 if SAGE (SQLSERVER_HOST) answers on its port
#
# Needs wireguard-tools (wg-quick) and sudo on this machine.
set -euo pipefail

need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "Missing $v in .env" >&2; exit 1; }; done; }
reachable() { ping -c1 -W2 "$1" >/dev/null 2>&1; }
wg_dir() { if [ "$(uname)" = Darwin ]; then echo "$(brew --prefix)/etc/wireguard"; else echo /etc/wireguard; fi; }
# The tunnel make brought up: its config exists and its interface is up (no sudo needed).
managed() { [ -f "$(wg_dir)/prike.conf" ] && { ip link show prike >/dev/null 2>&1 || [ "$(uname)" = Darwin ]; }; }
# The interface traffic to $1 leaves by (wg tunnel "prike", or eth0 under WSL
# when the WireGuard app on Windows carries it).
route_dev() { ip route get "$1" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1; }
is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }
# WireGuard drops a handshake whose timestamp is older than the last one the
# server accepted for this key, so a WSL clock behind Windows (common after
# sleep) makes the server ignore us. Seconds WSL is behind Windows (0 elsewhere).
clock_skew() {
  is_wsl || { echo 0; return; }
  local win; win=$(powershell.exe -NoProfile -Command '[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()' 2>/dev/null | tr -d '\r')
  [[ "$win" =~ ^[0-9]+$ ]] && echo $((win - $(date -u +%s))) || echo 0
}
via() {
  local dev; dev=$(route_dev "$1")
  if managed; then echo "the tunnel make manages (prike)"
  elif is_wsl; then echo "Windows (${dev:-?}): the WireGuard app there, not make"
  else echo "${dev:-another route}, not a tunnel make manages"; fi
}

case "${1:-}" in
  vpn-up)
    need PRIKE_WG_PRIVATE_KEY PRIKE_WG_ADDRESS PRIKE_WG_PEER_PUBLIC_KEY PRIKE_WG_ENDPOINT PRIKE_WG_ALLOWED_IPS PRIKE_HOST
    if reachable "$PRIKE_HOST"; then
      echo "Already reachable: $PRIKE_HOST answers via $(via "$PRIKE_HOST")."
      if ! managed && is_wsl; then
        echo "To manage it from WSL, turn the Windows tunnel off first (README: Prike tunnel from WSL), then make vpn-up again."
      fi
      exit 0
    fi
    command -v wg-quick >/dev/null || { echo "Install wireguard-tools first (apt/dnf/pacman/brew)." >&2; exit 1; }
    skew=$(clock_skew)
    if [ "${skew#-}" -gt 5 ]; then
      echo "WSL clock is ${skew}s off Windows: resyncing it (hwclock -s)."
      sudo hwclock -s 2>/dev/null || echo "hwclock failed: run 'wsl --shutdown' in Windows and reopen WSL." >&2
    fi
    # Resolve the endpoint to IPv4 here: under WSL (NAT) a name that also
    # has an IPv6 address can make WireGuard pick a route that goes nowhere.
    ep_host=${PRIKE_WG_ENDPOINT%:*}; ep_port=${PRIKE_WG_ENDPOINT##*:}
    ep_ip=$(getent ahostsv4 "$ep_host" | awk 'NR==1 {print $1}')
    [ -n "$ep_ip" ] || { echo "Cannot resolve $ep_host: check your Internet / DNS." >&2; exit 1; }
    tmp=$(umask 077; mktemp)
    {
      echo "[Interface]"
      echo "PrivateKey = $PRIKE_WG_PRIVATE_KEY"
      echo "Address = $PRIKE_WG_ADDRESS"
      echo
      echo "[Peer]"
      echo "PublicKey = $PRIKE_WG_PEER_PUBLIC_KEY"
      [ -n "${PRIKE_WG_PRESHARED_KEY:-}" ] && echo "PresharedKey = $PRIKE_WG_PRESHARED_KEY"
      echo "Endpoint = $ep_ip:$ep_port"
      echo "AllowedIPs = $PRIKE_WG_ALLOWED_IPS"
      echo "PersistentKeepalive = 25"
    } > "$tmp"
    sudo install -d -m 700 "$(wg_dir)"
    sudo install -m 600 "$tmp" "$(wg_dir)/prike.conf"
    rm -f "$tmp"
    sudo wg-quick down prike >/dev/null 2>&1 || true
    sudo wg-quick up prike >/dev/null
    echo "Tunnel up to $ep_host ($ep_ip:$ep_port); waiting for $PRIKE_HOST (up to 30 s)..."
    for _ in $(seq 1 10); do
      reachable "$PRIKE_HOST" && { echo "VPN up: $PRIKE_HOST answers."; exit 0; }
      sleep 1
    done
    {
      echo "The tunnel is up but $PRIKE_HOST does not answer. What WireGuard sees:"
      sudo wg show prike | grep -E 'endpoint|latest handshake|transfer' || true
      is_wsl && echo "  WSL clock vs Windows: $(clock_skew)s"
      echo "No 'latest handshake' and 0 B received: the server is not answering our UDP"
      echo "(another device using this same key, e.g. the Windows tunnel still on, or a firewall)."
      echo "A handshake but no ping: the server side; send this output to Asterio."
    } >&2
    exit 1 ;;

  vpn-down)
    if managed; then sudo wg-quick down prike; echo "Tunnel down."; exit 0; fi
    echo "No tunnel managed by make here."
    if [ -n "${PRIKE_HOST:-}" ] && reachable "$PRIKE_HOST"; then
      echo "$PRIKE_HOST still answers via $(via "$PRIKE_HOST"). Turn it off there."
    fi ;;

  vpn-status)
    need PRIKE_HOST
    if reachable "$PRIKE_HOST"; then echo "Up: $PRIKE_HOST answers via $(via "$PRIKE_HOST")."
    else echo "Down: run make vpn-up." >&2; exit 1; fi ;;

  ssh-setup)
    need PRIKE_HOST PRIKE_USER PRIKE_SSH_KEY_B64 PRIKE_HOST_KEY
    install -d -m 700 "$HOME/.ssh"
    (umask 077; printf '%s' "$PRIKE_SSH_KEY_B64" | base64 -d > "$HOME/.ssh/id_ed25519_prike")
    touch "$HOME/.ssh/known_hosts" "$HOME/.ssh/config"
    chmod 600 "$HOME/.ssh/config"
    grep -qF "$PRIKE_HOST_KEY" "$HOME/.ssh/known_hosts" || echo "$PRIKE_HOST_KEY" >> "$HOME/.ssh/known_hosts"
    if ! grep -qE '^Host prike$' "$HOME/.ssh/config"; then
      printf '\nHost prike\n    HostName %s\n    User %s\n    IdentityFile ~/.ssh/id_ed25519_prike\n    IdentitiesOnly yes\n' \
        "$PRIKE_HOST" "$PRIKE_USER" >> "$HOME/.ssh/config"
    fi
    echo "ssh prike is ready." ;;

  sage-check)
    need SQLSERVER_HOST
    host=${SQLSERVER_HOST%%,*}
    port=1433
    [ "$host" != "$SQLSERVER_HOST" ] && port=${SQLSERVER_HOST#*,}
    if ! [[ "$host" =~ ^[0-9.]+$ ]] && ! getent hosts "$host" >/dev/null; then
      echo "Cannot resolve $host: Windows machine names do not resolve on Linux/WSL." >&2
      echo "Put its IP in SQLSERVER_HOST (e.g. SQLSERVER_HOST=192.168.x.y,1433) or add it to /etc/hosts." >&2
      exit 1
    fi
    if timeout 5 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then
      echo "SAGE answers on $host:$port."
    else
      echo "SAGE ($host:$port) does not answer: bring up the VPN that reaches SAGE (make sage-vpn-up)." >&2
      exit 1
    fi ;;

  *)
    sed -n '2,13p' "$0"; exit 2 ;;
esac
