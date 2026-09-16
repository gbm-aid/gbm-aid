"""score_thresholds tablosuna donmus P33/P66 esiklerini yazar.

Kaynak: decisions/2026-08-18-omics-skor-formulleri-ve-esikler.md (KARAR 1).
48 hastalik molecular_scores kohortundan numpy dogrusal-interpolasyon P33/P66
hesaplanmis ve canli tmz_class/repair_class atamalarini 48/48 birebir
yeniden urettigi bagimsiz olarak dogrulanmistir (db-agent, 2026-08-18).

DAVRANIS:
  - Varsayilan: DRY-RUN (hicbir yazma yapilmaz, yalniz ne olacagi yazdirilir).
  - --apply: gercek INSERT (tek transaction, COMMIT).
  - Idempotent: (score_version, score_name) zaten varsa o satir ATLANIR
    (INSERT edilmez), score_thresholds'ta bu ciftin UNIQUE constraint'i
    olmadigi icin -- mukerrer yazimi burada, kod seviyesinde onluyoruz.
  - 2026-08-18 Codex capraz incelemesi BOSLUK 3 duzeltmesi: atlamadan
    ONCE mevcut satirin GECERLILIGI dogrulanir -- (1) p33_cutoff <
    p66_cutoff, (2) ikisi de [0,100] araliginda, (3) cohort_snapshot_n
    beklenen degerde (ROWS_TO_WRITE'daki ile ayni). Herhangi biri
    tutmazsa script SESSIZCE "zaten var" DEMEZ -- acik RuntimeError ile
    DURUR (--dry-run modunda bile, cunku bu bir onay adimi). Ayrica
    p33/p66 sayisal degerleri hedeften farkliysa (gecerli ama BASKA bir
    kalibrasyon) bu FATAL DEGIL ama acikca UYARI olarak yazdirilir --
    gelecekte bilincli bir versiyon guncellemesi (v1.1 vb.) bu yuzden
    engellenmesin diye.

Kullanim:
  python tools/write_score_thresholds.py            # dry-run
  python tools/write_score_thresholds.py --apply     # gercek yazim
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # gbm-aid mert/tools -> proje koku
ENV_PATH = PROJECT_ROOT / ".env"

# aggressiveness_score BILINCLI OLARAK YOK -- mimari (v45.txt satir 583-586)
# onu sabit mutlak esikli (<45/65/80) taniyor, score_thresholds'a girmiyor.
# Ayrica DB'nin kendi CHECK constraint'i (score_thresholds_score_name_check)
# bu ikisinden baskasini zaten reddediyor.
ROWS_TO_WRITE = [
    {
        "score_version": "v1.0",
        "score_name": "tmz_resistance_score",
        "p33_cutoff": 46.5009,
        "p66_cutoff": 48.3900,
        "cohort_snapshot_n": 48,
    },
    {
        "score_version": "v1.0",
        "score_name": "dna_repair_score",
        "p33_cutoff": 41.9847,
        "p66_cutoff": 59.0642,
        "cohort_snapshot_n": 48,
    },
]


def _load_env() -> dict:
    env = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def _connect(env: dict, readonly: bool):
    opts = "-c default_transaction_read_only=on" if readonly else ""
    return psycopg2.connect(
        host=env["SUPABASE_HOST"], port=env["SUPABASE_PORT"], dbname=env["SUPABASE_DB"],
        user=env["SUPABASE_USER"], password=env["SUPABASE_PASSWORD"],
        options=opts,
    )


class ExistingRowValidationError(RuntimeError):
    """(score_version, score_name) zaten var ama satırın kendisi bozuk --
    sessizce 'zaten var' denip atlanamaz."""


def _validate_existing_row(target: dict, existing_row: dict) -> list[str]:
    """existing_row: {'p33_cutoff':..., 'p66_cutoff':..., 'cohort_snapshot_n':...}
    Döndürür: FATAL sorunların listesi (boşsa satır geçerli)."""
    problems = []
    p33 = float(existing_row["p33_cutoff"])
    p66 = float(existing_row["p66_cutoff"])
    n = existing_row["cohort_snapshot_n"]

    if not (p33 < p66):
        problems.append(f"p33_cutoff({p33}) < p66_cutoff({p66}) DEĞİL")
    if not (0.0 <= p33 <= 100.0):
        problems.append(f"p33_cutoff({p33}) [0,100] aralığında DEĞİL")
    if not (0.0 <= p66 <= 100.0):
        problems.append(f"p66_cutoff({p66}) [0,100] aralığında DEĞİL")
    if n != target["cohort_snapshot_n"]:
        problems.append(
            f"cohort_snapshot_n({n}) beklenenden({target['cohort_snapshot_n']}) FARKLI"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="gercek INSERT calistir (varsayilan: dry-run)")
    args = parser.parse_args()

    env = _load_env()
    conn = _connect(env, readonly=not args.apply)
    cur = conn.cursor()

    cur.execute(
        "SELECT score_version, score_name, p33_cutoff, p66_cutoff, cohort_snapshot_n "
        "FROM score_thresholds"
    )
    existing_by_key = {
        (row[0], row[1]): {"p33_cutoff": row[2], "p66_cutoff": row[3], "cohort_snapshot_n": row[4]}
        for row in cur.fetchall()
    }
    existing = set(existing_by_key.keys())

    to_insert = [r for r in ROWS_TO_WRITE if (r["score_version"], r["score_name"]) not in existing]
    to_skip = [r for r in ROWS_TO_WRITE if (r["score_version"], r["score_name"]) in existing]

    print(f"Mevcut satirlar (score_version, score_name): {sorted(existing)}")
    print(f"YAZILACAK ({len(to_insert)}): {[(r['score_version'], r['score_name']) for r in to_insert]}")
    print(f"ATLANACAK -- ZATEN_VAR ({len(to_skip)}): {[(r['score_version'], r['score_name']) for r in to_skip]}")

    # Codex BOSLUK 3: atlamadan once mevcut satiri DOGRULA -- sessiz kabul YOK.
    for r in to_skip:
        key = (r["score_version"], r["score_name"])
        existing_row = existing_by_key[key]
        problems = _validate_existing_row(r, existing_row)
        if problems:
            cur.close()
            conn.close()
            raise ExistingRowValidationError(
                f"{key} zaten DB'de var AMA GECERSIZ -- sessizce 'zaten var' DENEMEZ:\n"
                f"  mevcut satir: {existing_row}\n"
                f"  sorunlar: {problems}\n"
                "  Bu, elle/baska bir yolla bozulmus bir score_thresholds satiri "
                "olabilir -- DUR, incele, gerekirse duzelt (bu script'in "
                "sorumlulugu DEGIL, kasitli-yeniden-kalibrasyon/duzeltme "
                "ayri onayli bir islem olmali)."
            )
        target_p33, target_p66 = r["p33_cutoff"], r["p66_cutoff"]
        actual_p33, actual_p66 = float(existing_row["p33_cutoff"]), float(existing_row["p66_cutoff"])
        if abs(actual_p33 - target_p33) > 1e-6 or abs(actual_p66 - target_p66) > 1e-6:
            print(
                f"  [UYARI] {key}: mevcut satir GECERLI ama bu modulun hedef "
                f"degerlerinden FARKLI -- mevcut p33={actual_p33}/p66={actual_p66}, "
                f"hedef p33={target_p33}/p66={target_p66}. Fatal DEGIL (belki "
                "bilincli bir versiyon guncellemesi), ama bu bir DRIFT sinyali "
                "olabilir -- Baris'a bildirin."
            )
        else:
            print(f"  [OK] {key}: mevcut satir gecerli VE hedef degerlerle birebir ayni.")

    if not args.apply:
        print("\n[DRY-RUN] --apply verilmedi, hicbir yazma yapilmadi.")
        cur.close()
        conn.close()
        return 0

    if not to_insert:
        print("\n[APPLY] Yazilacak yeni satir yok (hepsi zaten var), islem YOK.")
        cur.close()
        conn.close()
        return 0

    try:
        for r in to_insert:
            cur.execute(
                """
                INSERT INTO score_thresholds
                    (score_version, score_name, p33_cutoff, p66_cutoff, cohort_snapshot_n, frozen_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                """,
                (r["score_version"], r["score_name"], r["p33_cutoff"], r["p66_cutoff"], r["cohort_snapshot_n"]),
            )
        conn.commit()
        print(f"\n[APPLY] COMMIT edildi. {len(to_insert)} satir yazildi.")
    except Exception:
        conn.rollback()
        print("\n[APPLY] HATA -- ROLLBACK edildi.", file=sys.stderr)
        raise
    finally:
        cur.close()
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
