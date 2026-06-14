from database import get_connection

try:
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM support_tickets;")
    row_count = cur.fetchone()[0]
    
    cur.execute("SELECT ticket_id, customer_name, category FROM support_tickets LIMIT 1;")
    sample = cur.fetchone()
    
    print("Database Connection: SUCCESS")
    print(f"Total Rows Found: {row_count}")
    if sample:
        print(f"👀 Sample Ticket: ID={sample[0]}, Name={sample[1]}, Category={sample[2]}")
        
    cur.close()
    conn.close()
except Exception as e:
    print(f"Connection Failed: {e}")