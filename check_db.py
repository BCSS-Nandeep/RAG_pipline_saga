import sys
from db import get_pool
import psycopg.rows

pool = get_pool()
with pool.connection() as conn:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        # Get list of tables
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = [row['table_name'] for row in cur.fetchall()]
        print(f"Tables: {tables}")
        
        for t in ['social_media_posts', 'social_media_alerts', 'social_media_events']:
            if t in tables:
                cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{t}';")
                cols = cur.fetchall()
                print(f"\\nTable {t}:")
                for c in cols:
                    print(f"  {c['column_name']} ({c['data_type']})")
