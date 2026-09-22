import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_LOCAL = PROJECT_ROOT / ".env.local"
ENV = PROJECT_ROOT / ".env"

if ENV_LOCAL.exists():
    load_dotenv(ENV_LOCAL)
elif ENV.exists():
    load_dotenv(ENV)


DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_UNPOOLED = os.getenv("DATABASE_URL_UNPOOLED")
NEON_BRANCH = os.getenv("NEON_BRANCH")

NEON_S3_BUCKET = os.getenv("NEON_S3_BUCKET", "model-artifacts")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL_S3 = os.getenv("AWS_ENDPOINT_URL_S3")
AWS_REGION = os.getenv("AWS_REGION")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)


def validar_database():

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL não foi definida nas variáveis de ambiente."
        )


def validar_object_storage():

    required = {
        "NEON_S3_BUCKET": NEON_S3_BUCKET,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_ENDPOINT_URL_S3": AWS_ENDPOINT_URL_S3,
        "AWS_REGION": AWS_REGION
    }

    missing = [
        key
        for key, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            f"Variáveis de ambiente ausentes: {', '.join(missing)}"
        )
    