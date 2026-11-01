import sqlite3
from datetime import datetime, timedelta, timezone

DB_FILE = "contacts.db"
ROLES = ("authority", "resident")


def _conn():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        role TEXT NOT NULL DEFAULT 'resident',
        place TEXT NOT NULL DEFAULT '',
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS alerts_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lake_id TEXT NOT NULL,
        email TEXT NOT NULL,
        level TEXT NOT NULL,
        sent_at TEXT NOT NULL)""")
    return con


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_contact(name, email, role="resident", place=""):
    name, email = name.strip(), email.strip().lower()
    role, place = role.strip().lower(), place.strip()
    if not name:
        raise ValueError("name is required")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise ValueError("email does not look valid")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if role == "resident" and not place:
        raise ValueError("a resident needs a place (for example Chungthang)")
    con = _conn()
    try:
        cur = con.execute(
            "INSERT INTO contacts (name, email, role, place, created_at) VALUES (?,?,?,?,?)",
            (name, email, role, place, _now()))
        con.commit()
        return {"id": cur.lastrowid, "name": name, "email": email, "role": role, "place": place}
    except sqlite3.IntegrityError:
        raise ValueError("this email is already registered")
    finally:
        con.close()


def list_contacts(place=None):
    con = _conn()
    try:
        if place:
            rows = con.execute("SELECT * FROM contacts WHERE lower(place)=lower(?) ORDER BY id",
                               (place,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM contacts ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def delete_contact(contact_id):
    con = _conn()
    try:
        cur = con.execute("DELETE FROM contacts WHERE id=?", (contact_id,))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def already_sent(lake_id, email, level, hours):
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    con = _conn()
    try:
        row = con.execute(
            "SELECT 1 FROM alerts_log WHERE lake_id=? AND email=? AND level=? AND sent_at>=?",
            (lake_id, email, level, since)).fetchone()
        return row is not None
    finally:
        con.close()


def log_alert(lake_id, email, level):
    con = _conn()
    try:
        con.execute("INSERT INTO alerts_log (lake_id, email, level, sent_at) VALUES (?,?,?,?)",
                    (lake_id, email, level, _now()))
        con.commit()
    finally:
        con.close()
