"""GBM-AID PostgreSQL bağlantısının tek giriş noktası.

Bağlantı bilgileri yalnızca proje kökündeki ``.env`` dosyasından okunur.
Bu modüle veya çağıran scriptlere kimlik bilgisi yazılmamalıdır.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import connection


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"


def load_project_environment() -> None:
    """Proje kökündeki gizli ortam ayarlarını yükle."""

    if not ENV_FILE.is_file():
        raise FileNotFoundError(
            f"Bağlantı ayar dosyası bulunamadı: {ENV_FILE}. "
            ".env.example dosyasını örnek al."
        )
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def get_connection(*, readonly: bool = False) -> connection:
    """Yeni bir PostgreSQL bağlantısı aç."""

    load_project_environment()
    database_url = os.environ["DATABASE_URL"]
    db_connection = psycopg2.connect(
        database_url,
        application_name="gbm-aid-mert",
        connect_timeout=15,
        sslmode="require",
        # 2026-08-13 db-agent NOT: LUMIERE PyRadiomics --apply koşusunda
        # canlı olarak gözlemlendi -- Supabase pooler (aws-0-eu-west-3,
        # port 5432) bazen bağlantıyı sessizce (TCP RST/FIN GÖNDERMEDEN)
        # düşürüyor; keepalive'sız libpq bu durumda cursor.execute()/
        # commit()'te SÜRESİZ BLOKE OLUYOR (istisna fırlatmıyor, script'in
        # kendi satır-bazlı retry/reconnect mantığı bu yüzden hiç
        # tetiklenemiyor). TCP keepalive probe'ları ölü bağlantıyı ~20-25s
        # içinde tespit edip OperationalError fırlatmaya zorluyor, böylece
        # çağıran kod (ör. write_lumiere_pyradiomics_to_db.py'nin
        # reconnect-retry bloğu) süresiz asılı kalmak yerine gerçekten
        # tepki verebiliyor. Başarılı bağlantılar için davranış değişmiyor.
        keepalives=1,
        keepalives_idle=10,
        keepalives_interval=5,
        keepalives_count=3,
    )
    if readonly:
        db_connection.set_session(readonly=True, autocommit=True)
    return db_connection
