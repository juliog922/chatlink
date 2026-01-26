#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

import pyodbc


DEB_PATH = Path("/root/kapalua/chatlink/chatlink_bot/msodbcsql17_17.10.6.1-1_amd64.deb")
DRIVER_NAME = "ODBC Driver 17 for SQL Server"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("ACCEPT_EULA", "Y")
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    return subprocess.run(cmd, text=True, capture_output=True, env=env)


def driver_installed() -> bool:
    p = _run(["odbcinst", "-q", "-d"])
    if p.returncode != 0:
        return False
    return DRIVER_NAME.lower() in (p.stdout or "").lower()


def ensure_driver() -> None:
    if driver_installed():
        return

    if not DEB_PATH.exists():
        print(f"[ERROR] msodbcsql17 .deb not found at: {DEB_PATH}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Installing {DRIVER_NAME} from {DEB_PATH} ...")

    p = _run(["dpkg", "-i", str(DEB_PATH)])
    if p.returncode != 0:
        print("[WARN] dpkg -i failed, trying apt-get -f install to fix dependencies...")
        print(p.stderr.strip())
        p2 = _run(["apt-get", "-y", "-f", "install"])
        if p2.returncode != 0:
            print("[ERROR] apt-get -f install failed:", file=sys.stderr)
            print(p2.stderr.strip(), file=sys.stderr)
            sys.exit(3)

        p3 = _run(["dpkg", "-i", str(DEB_PATH)])
        if p3.returncode != 0:
            print("[ERROR] dpkg -i still failing:", file=sys.stderr)
            print(p3.stderr.strip(), file=sys.stderr)
            sys.exit(4)

    if not driver_installed():
        print(f"[ERROR] {DRIVER_NAME} still not registered in odbcinst.", file=sys.stderr)
        sys.exit(5)

    print("[INFO] Driver installed OK.")


def main() -> None:
    # Defaults per your message (override by exporting env vars if you want)
    user = os.getenv("SQLSERVER_USER", "BOT")
    password = os.getenv("SQLSERVER_PASSWORD", "fGVJ2Wasc@#")
    host = os.getenv("SQLSERVER_HOST", "192.168.1.242")
    db = os.getenv("SQLSERVER_DB", "Sage")
    port = os.getenv("SQLSERVER_PORT", "1433")

    if not password or password == "xxxxxxx":
        print('[ERROR] Set SQLSERVER_PASSWORD (don’t leave it as "xxxxxxx").', file=sys.stderr)
        sys.exit(6)

    ensure_driver()

    server = f"{host},{port}"
    conn_str = (
        f"DRIVER={{{DRIVER_NAME}}};"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={user};"
        f"PWD={password};"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=5;"
    )

    with pyodbc.connect(conn_str) as cn:
        cur = cn.cursor()

        cur.execute("SELECT COUNT(*) FROM Articulos;")
        total = int(cur.fetchone()[0])

        cur.execute("SELECT COUNT(*) FROM Articulos WHERE K_BOT = 1;")
        bot1 = int(cur.fetchone()[0])

        print(f"Total Articulos: {total}")
        print(f"Articulos with K_BOT=1: {bot1}")

        print("\nBreakdown by K_BOT:")
        cur.execute("SELECT K_BOT, COUNT(*) AS c FROM Articulos GROUP BY K_BOT ORDER BY K_BOT;")
        for k_bot, c in cur.fetchall():
            print(f"  K_BOT={k_bot!r}: {int(c)}")


if __name__ == "__main__":
    main()
