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

case "${1:-}" in
  vpn-up)
    need PRIKE_WG_PRIVATE_KEY PRIKE_WG_ADDRESS PRIKE_WG_PEER_PUBLIC_KEY PRIKE_WG_ENDPOINT PRIKE_WG_ALLOWED_IPS PRIKE_HOST
    if reachable "$PRIKE_HOST"; then echo "VPN already up: $PRIKE_HOST answers."; exit 0; fi
    command -v wg-quick >/dev/null || { echo "Install wireguard-tools first (apt/dnf/pacman/brew)." >&2; exit 1; }
    tmp=$(umask 077; mktemp)
    {
      echo "[Interface]"
      echo "PrivateKey = $PRIKE_WG_PRIVATE_KEY"
      echo "Address = $PRIKE_WG_ADDRESS"
      echo
      echo "[Peer]"
      echo "PublicKey = $PRIKE_WG_PEER_PUBLIC_KEY"
      [ -n "${PRIKE_WG_PRESHARED_KEY:-}" ] && echo "PresharedKey = $PRIKE_WG_PRESHARED_KEY"
      echo "Endpoint = $PRIKE_WG_ENDPOINT"
      echo "AllowedIPs = $PRIKE_WG_ALLOWED_IPS"
      echo "PersistentKeepalive = 25"
    } > "$tmp"
    sudo install -d -m 700 "$(wg_dir)"
    sudo install -m 600 "$tmp" "$(wg_dir)/prike.conf"
    rm -f "$tmp"
    sudo wg-quick down prike >/dev/null 2>&1 || true
    sudo wg-quick up prike >/dev/null
    for _ in $(seq 1 15); do
      reachable "$PRIKE_HOST" && { echo "VPN up: $PRIKE_HOST answers."; exit 0; }
      sleep 2
    done
    echo "Tunnel is up but $PRIKE_HOST does not answer. Check your Internet, then ask Asterio." >&2
    exit 1 ;;

  vpn-down)
    sudo wg-quick down prike ;;

  vpn-status)
    need PRIKE_HOST
    if reachable "$PRIKE_HOST"; then echo "VPN up."; else echo "VPN down: run make vpn-up." >&2; exit 1; fi ;;

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
    if timeout 5 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then
      echo "SAGE answers on $host:$port."
    else
      echo "SAGE ($host:$port) does not answer: bring up the VPN that reaches SAGE (make sage-vpn-up)." >&2
      exit 1
    fi ;;

  *)
    sed -n '2,13p' "$0"; exit 2 ;;
esac
