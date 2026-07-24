import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import redis

load_dotenv()

app = FastAPI(title="GBM-AID API")

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")


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
