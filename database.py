import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT")
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_table():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id         SERIAL PRIMARY KEY,
            timestamp         TIMESTAMP,
            customer_name     VARCHAR(100),
            category          VARCHAR(50),
            issue_description TEXT,
            rating            INT,
            order_value       NUMERIC(10, 2),
            status            VARCHAR(20),
            resolution_eta    VARCHAR(50),
            device_type       VARCHAR(30)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Table 'support_tickets' is ready.")


def fetch_all_tickets() -> pd.DataFrame:
    """
    Fetches every row from support_tickets and returns a Pandas DataFrame.

    BUG FIXED: Original code had a standalone `df` line (dead code) before
    closing the connection. Removed.
    """
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM support_tickets ORDER BY ticket_id DESC;", conn
    )
    conn.close()
    return df


def fetch_tickets_by_category(category: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM support_tickets WHERE category = %s ORDER BY ticket_id DESC;",
        conn, params=(category,)
    )
    conn.close()
    return df
