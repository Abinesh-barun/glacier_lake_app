"""
Email alert system: database helpers + sender + LangChain agent (ONE tool).

Parts:
  1. DATABASE   - list / add / delete subscribers in subscribers.db
  2. SENDER     - send_one_email() and deliver_to_all()
  3. AGENT      - Groq LLM writes the alert, then calls ONE tool to send it
  4. FALLBACK   - send_plain_alert() works without any LLM / Groq key

Safe by default: if SMTP_USER / SMTP_PASSWORD are not set, emails are only
PRINTED (dry run) instead of sent. Put these in your .env to really send:
    GROQ_API_KEY=...
    SMTP_USER=youraddress@gmail.com
    SMTP_PASSWORD=your-gmail-app-password
"""
import os
import re
import smtplib
import sqlite3
from email.message import EmailMessage

from dotenv import load_dotenv
load_dotenv()

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subscribers.db")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ============================================================ 1. DATABASE
def _connect():
    """Open the database and make sure the table exists."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS subscribers (
                        id    INTEGER PRIMARY KEY,
                        name  TEXT NOT NULL,
                        email TEXT NOT NULL)""")
    return conn


def list_subscribers():
    """Return a list of (id, name, email)."""
    conn = _connect()
    rows = conn.execute("SELECT id, name, email FROM subscribers ORDER BY id").fetchall()
    conn.close()
    return rows


def get_subscribers():
    """Return a list of (name, email)  - used when sending."""
    return [(name, email) for _, name, email in list_subscribers()]


def add_subscriber(name, email):
    """Add one person. Returns (True/False, message)."""
    name, email = name.strip(), email.strip().lower()
    if not name:
        return False, "Please enter a name."
    if not EMAIL_PATTERN.match(email):
        return False, "That email address does not look valid."
    conn = _connect()
    exists = conn.execute("SELECT 1 FROM subscribers WHERE lower(email)=?", (email,)).fetchone()
    if exists:
        conn.close()
        return False, f"{email} is already subscribed."
    conn.execute("INSERT INTO subscribers (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    conn.close()
    return True, f"Added {name}."


def delete_subscriber(sub_id):
    conn = _connect()
    conn.execute("DELETE FROM subscribers WHERE id=?", (sub_id,))
    conn.commit()
    conn.close()


# ============================================================ 2. SENDER
def smtp_configured():
    return bool(os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASSWORD"))


def send_one_email(to_name, to_email, subject, body):
    """Send one email. Prints instead of sending if SMTP is not configured."""
    if not smtp_configured():
        print(f"[DRY RUN] to {to_name} <{to_email}> | {subject}")
        return

    user, password = os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"]
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(f"Hello {to_name},\n\n{body}\n\n- Glacial Lake Monitoring")

    with smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                      int(os.environ.get("SMTP_PORT", 587))) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)


def deliver_to_all(subject, message):
    """Email every subscriber. Returns a short confirmation text."""
    people = get_subscribers()
    if not people:
        return "No subscribers in the database - nothing was sent."
    for name, email in people:
        send_one_email(name, email, subject, message)
    mode = "" if smtp_configured() else " (dry run - SMTP not set, nothing really sent)"
    return f"Alert sent to {len(people)} subscribers{mode}."


# ============================================================ 3. AGENT
def build_alert_agent():
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq

    @tool
    def send_alert_emails(subject: str, message: str) -> str:
        """Email an alert to EVERY subscriber in the database.
        subject: short email subject line.
        message: the alert text (no greeting; the name is added automatically)."""
        return deliver_to_all(subject, message)

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return create_agent(
        llm,
        tools=[send_alert_emails],
        system_prompt=(
            "You are a glacier early-warning assistant. You get a measurement "
            "summary. Write a short, clear alert (subject + 3-4 sentence "
            "message) using ONLY the numbers given, then call the "
            "send_alert_emails tool exactly once. Afterwards reply with one "
            "sentence confirming what was sent."
        ),
    )


def _facts_text(facts):
    return "\n".join(f"- {k}: {v}" for k, v in facts.items())


def send_risk_alert(facts, subject_area="Glacial lake"):
    """Agent version: the LLM writes the email and sends it with the tool."""
    out = build_alert_agent().invoke({"messages": [{
        "role": "user",
        "content": f"Subject area: {subject_area}\nMeasurements:\n{_facts_text(facts)}\n\nSend the alert."}]})
    return out["messages"][-1].content


# ============================================================ 4. FALLBACK
def send_plain_alert(facts, subject_area="Glacial lake"):
    """No-LLM version: fixed message built from the numbers."""
    subject = f"Glacier alert: {subject_area.lower()} change is {facts['risk']}"
    message = (f"The {subject_area.lower()} area is {facts['status'].lower()} "
               f"({facts['change_percent']} between the two images).\n"
               f"Risk level: {facts['risk']} (score {facts['risk_score']}).\n"
               "Please review the dashboard for details.")
    return deliver_to_all(subject, message)


if __name__ == "__main__":
    demo = {"change_percent": "+18.40%", "status": "Increasing", "risk": "Risky",
            "risk_score": "1.8 out of 10", "area_first": "4.10% of the image",
            "area_second": "4.85% of the image"}
    print(send_risk_alert(demo) if os.environ.get("GROQ_API_KEY") else send_plain_alert(demo))