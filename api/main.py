import os
from fastapi import FastAPI
from dotenv import load_dotenv
import psycopg2
import redis

load_dotenv()

app = FastAPI(title="GBM-AID API")

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")


@app.get("/")
def read_root():
    return {"message": "GBM-AID API calisiyor"}


@app.get("/health")
def health_check():
    status = {"supabase": "error", "redis": "error"}

    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
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
