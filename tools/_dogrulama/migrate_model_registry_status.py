"""
model_registry migrasyonu:
  1) status TEXT NOT NULL DEFAULT 'shadow' + CHECK (shadow|production|retired)
  2) UNIQUE (model_name, version)
  3) Kismi UNIQUE index: her model_name icin ayni anda en fazla 1 'production' satiri

Varsayilan (bayraksiz) calisma = DRY-RUN: DDL AYNI TRANSACTION icinde calistirilir
(gercek hata varsa burada yakalanir), sonra ICINDEKI SELECT'lerle dogrulanir,
sonra ROLLBACK yapilir -- DB'de HICBIR KALICI DEGISIKLIK OLMAZ.

--apply verilirse: ayni adimlar calisir ama sonunda COMMIT edilir.
"""
import sys
import psycopg2

ENV_PATH = r"C:\Users\Barış\Desktop\GBM-AID Prototip\.env"

DDL_STATEMENTS = [
    "ALTER TABLE model_registry ADD COLUMN status TEXT NOT NULL DEFAULT 'shadow';",
    "ALTER TABLE model_registry ADD CONSTRAINT model_registry_status_check "
    "CHECK (status IN ('shadow', 'production', 'retired'));",
    "ALTER TABLE model_registry ADD CONSTRAINT model_registry_model_name_version_uniq "
    "UNIQUE (model_name, version);",
    "CREATE UNIQUE INDEX model_registry_one_production_per_model "
    "ON model_registry (model_name) WHERE status = 'production';",
]


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


def pre_check(cur):
    print("=== ON KONTROL (migrasyondan ONCE, ayri sorgu) ===")
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'model_registry' AND column_name = 'status';
    """)
    print("status kolonu var mi (bos liste = yok):", cur.fetchall())

    cur.execute("""
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'model_registry'::regclass AND contype = 'u';
    """)
    print("UNIQUE constraint'ler (bos liste = yok):", cur.fetchall())

    cur.execute("SELECT COUNT(*) FROM model_registry;")
    print("satir sayisi:", cur.fetchone())


def apply_ddl(cur):
    print("\n=== DDL CALISTIRILIYOR (ayni transaction icinde) ===")
    for stmt in DDL_STATEMENTS:
        print(">>", stmt)
        cur.execute(stmt)
    print("DDL hatasiz calisti.")


def post_check(cur):
    print("\n=== TRANSACTION-ICI DOGRULAMA (henuz commit/rollback edilmedi) ===")
    cur.execute("""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'model_registry' AND column_name = 'status';
    """)
    print("status kolonu:", cur.fetchall())

    cur.execute("""
        SELECT conname, contype, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'model_registry'::regclass
        ORDER BY conname;
    """)
    print("tum constraint'ler:")
    for row in cur.fetchall():
        print("  ", row)

    cur.execute("""
        SELECT indexname, indexdef FROM pg_indexes
        WHERE tablename = 'model_registry' ORDER BY indexname;
    """)
    print("tum index'ler:")
    for row in cur.fetchall():
        print("  ", row)

    cur.execute("SELECT COUNT(*) FROM model_registry;")
    print("satir sayisi (hala 0 olmali):", cur.fetchone())


def main():
    apply_mode = "--apply" in sys.argv
    env = load_env()
    conn = psycopg2.connect(env["DATABASE_URL"])
    conn.autocommit = False
    cur = conn.cursor()

    try:
        pre_check(cur)
        apply_ddl(cur)
        post_check(cur)

        if apply_mode:
            conn.commit()
            print("\n=== COMMIT EDILDI (--apply) ===")
        else:
            conn.rollback()
            print("\n=== ROLLBACK EDILDI (dry-run, --apply verilmedi -- HICBIR KALICI DEGISIKLIK YOK) ===")
    except Exception as e:
        conn.rollback()
        print("\n=== HATA, ROLLBACK EDILDI ===")
        print(repr(e))
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
