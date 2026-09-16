"""
Fonksiyonel test - yeni constraint'lerin GERCEKTEN calistigini dogrular.
Her test kendi SAVEPOINT'i icinde calisir, sonunda tum transaction
ROLLBACK edilir -- model_registry 0 satirda kalir.
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


def try_case(cur, name, sql, params, expect_fail):
    cur.execute("SAVEPOINT sp;")
    try:
        cur.execute(sql, params)
        cur.execute("RELEASE SAVEPOINT sp;")
        ok = not expect_fail
        print(f"[{'OK' if ok else 'BEKLENMEDIK BASARI'}] {name} -> INSERT basarili (hata beklenmiyordu: {not expect_fail})")
    except Exception as e:
        cur.execute("ROLLBACK TO SAVEPOINT sp;")
        ok = expect_fail
        print(f"[{'OK' if ok else 'BEKLENMEDIK HATA'}] {name} -> reddedildi: {type(e).__name__}: {str(e).splitlines()[0]}")


def main():
    env = load_env()
    conn = psycopg2.connect(env["DATABASE_URL"])
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute("BEGIN;")

        # 1) Gecerli ilk kayit - shadow, basarili olmali
        try_case(
            cur, "1) gecerli shadow INSERT",
            "INSERT INTO model_registry (model_name, version, status) VALUES (%s,%s,%s)",
            ("cox_phm", "vTEST.0", "shadow"),
            expect_fail=False,
        )

        # 2) Ayni (model_name, version) tekrar INSERT - UNIQUE ihlali beklenir
        try_case(
            cur, "2) ayni (model_name,version) tekrar INSERT",
            "INSERT INTO model_registry (model_name, version, status) VALUES (%s,%s,%s)",
            ("cox_phm", "vTEST.0", "shadow"),
            expect_fail=True,
        )

        # 3) Gecersiz status degeri - CHECK ihlali beklenir
        try_case(
            cur, "3) gecersiz status ('deployed')",
            "INSERT INTO model_registry (model_name, version, status) VALUES (%s,%s,%s)",
            ("cox_phm", "vTEST.1", "deployed"),
            expect_fail=True,
        )

        # 4) Ayni model_name icin IKINCI production satiri - kismi UNIQUE index ihlali beklenir
        try_case(
            cur, "4a) ilk production satiri (cox_phm vTEST.2)",
            "INSERT INTO model_registry (model_name, version, status) VALUES (%s,%s,%s)",
            ("cox_phm", "vTEST.2", "production"),
            expect_fail=False,
        )
        try_case(
            cur, "4b) ikinci production satiri AYNI model_name (cox_phm vTEST.3)",
            "INSERT INTO model_registry (model_name, version, status) VALUES (%s,%s,%s)",
            ("cox_phm", "vTEST.3", "production"),
            expect_fail=True,
        )

        # 5) status verilmezse DEFAULT 'shadow' calisir mi
        cur.execute("SAVEPOINT sp;")
        cur.execute(
            "INSERT INTO model_registry (model_name, version) VALUES (%s,%s) RETURNING status",
            ("xgboost", "vTEST.0"),
        )
        row = cur.fetchone()
        cur.execute("RELEASE SAVEPOINT sp;")
        print(f"[{'OK' if row[0] == 'shadow' else 'BEKLENMEDIK'}] 5) status verilmeden INSERT -> default = {row[0]!r}")

        print("\nTum test satirlari transaction icinde, ROLLBACK ediliyor (kalici degisiklik yok)...")
    finally:
        conn.rollback()
        cur.execute("SELECT COUNT(*) FROM model_registry;")
        print("ROLLBACK sonrasi satir sayisi (0 olmali):", cur.fetchone())
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
