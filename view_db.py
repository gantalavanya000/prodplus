import sqlite3
import os

DB_PATH = 'database.db'

def view_data():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        print("\n" + "="*30)
        print("TABLE: users")
        print("="*30)
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        if not rows:
            print("No data found in users table.")
        for row in rows:
            # Mask password for display
            d = dict(row)
            if 'password' in d:
                d['password'] = '****'
            print(d)

        print("\n" + "="*30)
        print("TABLE: responses")
        print("="*30)
        cursor.execute("SELECT * FROM responses ORDER BY submission_date DESC LIMIT 10")
        rows = cursor.fetchall()
        if not rows:
            print("No data found in responses table.")
        for row in rows:
            print(dict(row))
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    view_data()
