import sqlite3

c = sqlite3.connect("subscribers.db")
c.execute("UPDATE subscribers SET name=?, email=? WHERE id=1", ("Abinesh", "abineshmass368@gmail.com"))
c.execute("DELETE FROM subscribers WHERE id != 1")
c.commit()

# show the table so you can see the change
for row in c.execute("SELECT * FROM subscribers"):
    print(row)
c.close()