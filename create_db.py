"""
Step 1 - Create the subscribers database.
Run once:  python create_db.py
Makes a file called subscribers.db with a table of 10 dummy people.
"""
import sqlite3

DB_FILE = "subscribers.db"

PEOPLE = [
    (1, "Aarav Sharma",   "aarav.sharma@example.com"),
    (2, "Diya Patel",     "diya.patel@example.com"),
    (3, "Rohan Verma",    "rohan.verma@example.com"),
    (4, "Ananya Iyer",    "ananya.iyer@example.com"),
    (5, "Karthik Raj",    "karthik.raj@example.com"),
    (6, "Meera Nair",     "meera.nair@example.com"),
    (7, "Vikram Singh",   "vikram.singh@example.com"),
    (8, "Sneha Reddy",    "sneha.reddy@example.com"),
    (9, "Arjun Menon",    "arjun.menon@example.com"),
    (10, "Priya Das",     "priya.das@example.com"),
]

conn = sqlite3.connect(DB_FILE)
conn.execute("DROP TABLE IF EXISTS subscribers")
conn.execute("""
    CREATE TABLE subscribers (
        id    INTEGER PRIMARY KEY,
        name  TEXT NOT NULL,
        email TEXT NOT NULL
    )
""")
conn.executemany("INSERT INTO subscribers VALUES (?, ?, ?)", PEOPLE)
conn.commit()

print("id | name | email")
for row in conn.execute("SELECT * FROM subscribers"):
    print(" | ".join(str(x) for x in row))
conn.close()
