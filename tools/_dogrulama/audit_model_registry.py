import sys
sys.path.insert(0, r"C:\Users\Barış\Desktop\GBM-AID Prototip\gbm-aid mert")
from db_connection import get_connection
import psycopg2.extras

conn = get_connection(readonly=True)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

print("=== COLUMNS ===")
cur.execute("""
    SELECT column_name, data_type, udt_name, is_nullable, column_default,
           character_maximum_length
    FROM information_schema.columns
    WHERE table_name = 'model_registry'
    ORDER BY ordinal_position
""")
for row in cur.fetchall():
    print(row)

print("=== CONSTRAINTS ===")
cur.execute("""
    SELECT conname, contype, pg_get_constraintdef(oid) AS def
    FROM pg_constraint
    WHERE conrelid = 'model_registry'::regclass
""")
for row in cur.fetchall():
    print(row)

print("=== ROW COUNT ===")
cur.execute("SELECT COUNT(*) AS n FROM model_registry")
print(cur.fetchone())

print("=== INDEXES ===")
cur.execute("""
    SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'model_registry'
""")
for row in cur.fetchall():
    print(row)

cur.close()
conn.close()
