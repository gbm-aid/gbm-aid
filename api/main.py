import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import redis
from pipeline.harmonization import (
    apply_n4_bias_correction,
    apply_zscore_normalization,
    apply_combat_harmonization,
    get_segmentation_mask,
)

load_dotenv()

app = FastAPI(title="GBM-AID API")

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

REQUIRED_MODALITIES = {"T1", "T1ce", "T2", "FLAIR"}


def get_db_connection():
    return psycopg2.connect(DATABASE_URL, connect_timeout=5)


@app.get("/")
def read_root():
    return {"message": "GBM-AID API calisiyor"}


@app.get("/health")
def health_check():
    status = {"supabase": "error", "redis": "error"}

    try:
        conn = get_db_connection()
        conn.close()
        status["supabase"] = "ok"
    except Exception as e:
        status["supabase"] = f"error: {str(e)}"

    try:
        r = redis.from_url(REDIS_URL, socket_connect_timeout=5)
        r.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"

    return status


@app.get("/patient/{patient_id}/status")
def get_patient_status(patient_id: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            "SELECT patient_id, scan_date, modality, harmonization_status "
            "FROM mr_scans WHERE patient_id = %s ORDER BY scan_date",
            (patient_id,),
        )
        scans = cur.fetchall()
        cur.close()
        conn.close()

        if not scans:
            raise HTTPException(
                status_code=404,
                detail=f"Hasta bulunamadi veya tarama kaydi yok: {patient_id}",
            )

        return {"patient_id": patient_id, "scans": scans}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Veritabani hatasi: {str(e)}"
        )


@app.post("/patient/{patient_id}/harmonize")
def harmonize_patient(patient_id: str):
    """
    Harmonizasyon pipeline'ini bir hasta icin calistirir:
    N4 -> Z-score -> ComBat -> segmentasyon.
    Mert/Nisa fonksiyonlari doldurulduktan sonra bu endpoint gercek
    veriyle uctan uca calisacak. Su an stub'lar NotImplementedError firlatir,
    bu da 501 olarak donuyor (henuz implemente edilmedi anlaminda).
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT modality, file_path FROM mr_scans WHERE patient_id = %s",
            (patient_id,),
        )
        scans = cur.fetchall()
        cur.close()
        conn.close()

        if not scans:
            raise HTTPException(
                status_code=404,
                detail=f"Hasta bulunamadi: {patient_id}",
            )

        # Eksik modalite kontrolu
        available = {s["modality"] for s in scans}
        missing = REQUIRED_MODALITIES - available
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Eksik modalite(ler): {sorted(missing)}. "
                f"Gerekli: {sorted(REQUIRED_MODALITIES)}",
            )

        results = {}
        for scan in scans:
            try:
                n4_path = apply_n4_bias_correction(scan["file_path"])
                zscore_path = apply_zscore_normalization(n4_path, source="unknown")
                results[scan["modality"]] = {"status": "ok", "path": zscore_path}
            except NotImplementedError:
                results[scan["modality"]] = {
                    "status": "pending",
                    "detail": "Harmonizasyon fonksiyonu henuz implemente edilmedi",
                }
            except FileNotFoundError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Bozuk veya bulunamayan dosya: {scan['file_path']}",
                )
            except TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Harmonizasyon zaman asimina ugradi: {scan['modality']}",
                )

        return {"patient_id": patient_id, "harmonization_results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Beklenmeyen hata: {str(e)}"
        )
