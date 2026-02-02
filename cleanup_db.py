import sqlite3

DB_NAME = 'database.db'

def cleanup():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Delete responses for csv users
    c.execute("DELETE FROM responses WHERE user_id IN (SELECT id FROM users WHERE username LIKE 'csv_user_%')")
    
    # Delete the users
    c.execute("DELETE FROM users WHERE username LIKE 'csv_user_%'")
    
    print("Deleted all csv_user records and their responses via pattern match.")
        
    conn.commit()
    conn.close()

if __name__ == '__main__':
    cleanup()
