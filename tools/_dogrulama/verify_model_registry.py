"""
SALT-OKUNUR dogrulama - model_registry tablosu.
Hicbir ALTER/INSERT/UPDATE calistirmiyor. Sadece bilgi_semasi/pg_catalog SELECT.
"""

# 2026-09-16: sabit kodlanmis mutlak yollar KALDIRILDI -- kullanici dizini
# adi halka acik depoya siziyordu ve bu script baska makinede calismiyordu.
# Yollar artik dosyanin kendi konumundan / ortam degiskeninden turetilir.
import os as _os
from pathlib import Path as _P
_BURASI   = _P(__file__).resolve().parent          # tools/_dogrulama
_KOK_KOD  = _P(__file__).resolve().parents[2]      # gbm-aid mert
_KOK_PROJ = _P(__file__).resolve().parents[3]      # GBM-AID Prototip
_VERI_KOKU = _P(_os.environ.get('GBMAID_VERI_KOKU', str(_KOK_PROJ)))
import psycopg2

ENV_PATH = str(_KOK_PROJ / '.env')


def load_env():
    env = {}
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def main():
    env = load_env()
    dsn = env["DATABASE_URL"]
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    print("=== 1. KOLONLAR (information_schema.columns) ===")
    cur.execute("""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'model_registry'
        ORDER BY ordinal_position;
    """)
    for row in cur.fetchall():
        print(row)

    print("\n=== 2. CONSTRAINT'LER (table_constraints + check_constraints) ===")
    cur.execute("""
        SELECT tc.constraint_name, tc.constraint_type, cc.check_clause
        FROM information_schema.table_constraints tc
        LEFT JOIN information_schema.check_constraints cc
          ON tc.constraint_name = cc.constraint_name
        WHERE tc.table_name = 'model_registry';
    """)
    for row in cur.fetchall():
        print(row)

    print("\n=== 3. UNIQUE / PK kolon detayi (key_column_usage) ===")
    cur.execute("""
        SELECT tc.constraint_name, tc.constraint_type, kcu.column_name, kcu.ordinal_position
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = 'model_registry'
        ORDER BY tc.constraint_name, kcu.ordinal_position;
    """)
    for row in cur.fetchall():
        print(row)

    print("\n=== 4. INDEX'ler (pg_indexes) ===")
    cur.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'model_registry';
    """)
    for row in cur.fetchall():
        print(row)

    print("\n=== 5. SATIR SAYISI ===")
    cur.execute("SELECT COUNT(*) FROM model_registry;")
    print(cur.fetchone())

    print("\n=== 6. TUM SATIRLAR (varsa) ===")
    cur.execute("SELECT * FROM model_registry;")
    for row in cur.fetchall():
        print(row)

    print("\n=== 7. Tablo yorumu / sequence sahibi (nextval dogrulamasi) ===")
    cur.execute("""
        SELECT column_name, column_default
        FROM information_schema.columns
        WHERE table_name = 'model_registry' AND column_default LIKE 'nextval%';
    """)
    for row in cur.fetchall():
        print(row)

    print("\n=== 8. 13 tablo listesi (sema genel kontrolu) ===")
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    for row in cur.fetchall():
        print(row)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
