"""SALT-OKUNUR - model_registry'ye bagimli nesneleri (view/trigger/FK) kontrol eder."""

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
    conn = psycopg2.connect(env["DATABASE_URL"])
    cur = conn.cursor()

    print("=== Views referencing model_registry ===")
    cur.execute("""
        SELECT table_name FROM information_schema.view_table_usage
        WHERE table_name = 'model_registry';
    """)
    print(cur.fetchall())

    print("=== FKs referencing model_registry (as target) ===")
    cur.execute("""
        SELECT tc.table_name, tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
        WHERE ccu.table_name = 'model_registry' AND tc.constraint_type = 'FOREIGN KEY';
    """)
    print(cur.fetchall())

    print("=== Triggers on model_registry ===")
    cur.execute("""
        SELECT trigger_name, event_manipulation FROM information_schema.triggers
        WHERE event_object_table = 'model_registry';
    """)
    print(cur.fetchall())

    print("=== RLS policies on model_registry ===")
    cur.execute("""
        SELECT polname FROM pg_policy p
        JOIN pg_class c ON p.polrelid = c.oid
        WHERE c.relname = 'model_registry';
    """)
    print(cur.fetchall())

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
