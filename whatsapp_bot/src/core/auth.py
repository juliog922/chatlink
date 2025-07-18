import sqlite3
import hashlib
import getpass
import logging


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_credentials() -> bool:
    logging.info("Authentication required")
    user = input("Phone: ").strip()
    password = getpass.getpass("Password: ").strip()

    conn = sqlite3.connect("auth.db")
    cur = conn.cursor()
    cur.execute("SELECT password, role FROM users WHERE phone = ?", (user,))
    row = cur.fetchone()
    conn.close()

    if not row:
        logging.error("User not found.")
        return False

    stored_hash, role = row
    if hash_password(password) != stored_hash:
        logging.error("Incorrect password.")
        return False

    if role != "admin":
        logging.error("Only the admin can log in from terminal.")
        return False

    logging.info("Authentication successful.")
    return True
