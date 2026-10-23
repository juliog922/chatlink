#!/usr/bin/env bash
# Helpers for the Prike server and SAGE, from variables in .env (git-ignored).
# Called by the Makefile:
#
#   prike.sh reach        exit 0 if the Prike server answers (tunnel up on Windows)
#   prike.sh ssh-setup    install the ssh key, the server's host key and the `prike` alias
#   prike.sh sage-check   exit 0 if SAGE (SQLSERVER_HOST) answers on its port
#
# The WireGuard tunnel itself runs on Windows: Asterio's conectar-prike.ps1.
set -euo pipefail

need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "Missing $v in .env" >&2; exit 1; }; done; }
reachable() { ping -c1 -W2 "$1" >/dev/null 2>&1; }
case "${1:-}" in
  reach)
    need PRIKE_HOST
    reachable "$PRIKE_HOST" && { echo "Prike answers ($PRIKE_HOST)."; exit 0; }
    echo "Prike ($PRIKE_HOST) does not answer: connect the tunnel on Windows (Asterio's conectar-prike.ps1, PowerShell as Administrator)." >&2
    exit 1 ;;

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
    sed -n '2,9p' "$0"; exit 2 ;;
esac
