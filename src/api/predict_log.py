# --------------------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------------------- #

from __future__ import annotations
from typing import Any

from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg

from src.config.neon import DATABASE_URL, validar_database

# from src.api.predict_log import registrar_predict

# --------------------------------------------------------------------------------------- #
# Instâncias
# --------------------------------------------------------------------------------------- #

SAO_PAULO = ZoneInfo("America/Sao_Paulo")


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_predict_requests (
    id BIGSERIAL PRIMARY KEY,

    requested_at TIMESTAMPTZ NOT NULL,

    status_code INTEGER NOT NULL,
    status_message TEXT,

    age INTEGER,
    job TEXT,
    marital TEXT,
    education TEXT,
    housing TEXT,
    loan TEXT,
    month TEXT,
    day_of_week TEXT,
    poutcome TEXT,
    month_position TEXT,
    year INTEGER,
    previous INTEGER,

    emp_var_rate DOUBLE PRECISION,
    cons_price_idx DOUBLE PRECISION,
    cons_conf_idx DOUBLE PRECISION,
    euribor3m DOUBLE PRECISION,
    nr_employed DOUBLE PRECISION,

    recommended_contact TEXT,
    conversion_probability DOUBLE PRECISION,
    accept_offer BOOLEAN,
    threshold DOUBLE PRECISION
)
"""

# --------------------------------------------------------------------------------------- #
# Funções
# --------------------------------------------------------------------------------------- #

def criar_tabela_predict_requests():
    validar_database()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)

        conn.commit()


def registrar_predict(payload, indicadores, resultado, status_code, status_message):
    try:
        validar_database()

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO api_predict_requests (
                        requested_at,
                        status_code,
                        status_message,
                        age,
                        job,
                        marital,
                        education,
                        housing,
                        loan,
                        month,
                        day_of_week,
                        poutcome,
                        month_position,
                        year,
                        previous,
                        emp_var_rate,
                        cons_price_idx,
                        cons_conf_idx,
                        euribor3m,
                        nr_employed,
                        recommended_contact,
                        conversion_probability,
                        accept_offer,
                        threshold
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        datetime.now(SAO_PAULO),
                        status_code,
                        status_message,
                        payload.get("age"),
                        payload.get("job"),
                        payload.get("marital"),
                        payload.get("education"),
                        payload.get("housing"),
                        payload.get("loan"),
                        payload.get("month"),
                        payload.get("day_of_week"),
                        payload.get("poutcome"),
                        payload.get("month_position"),
                        payload.get("year"),
                        payload.get("previous"),
                        indicadores.get("emp_var_rate"),
                        indicadores.get("cons_price_idx"),
                        indicadores.get("cons_conf_idx"),
                        indicadores.get("euribor3m"),
                        indicadores.get("nr_employed"),
                        resultado.get("recommended_contact"),
                        resultado.get("conversion_probability"),
                        resultado.get("accept_offer"),
                        resultado.get("threshold")
                    )
                )

            conn.commit()

    except Exception as exc:
        print("Erro ao persistir log de predict:", exc)


def processar_predict(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Executa a predição e registra a tentativa no Neon."""

    # Import local para evitar dependência circular.
    from src.api.api_endpoints import ValidationError, executar_predict

    indicadores = {}
    resultado = {}

    try:
        resultado, indicadores = executar_predict(payload)

        registrar_predict(
            payload=payload,
            indicadores=indicadores,
            resultado=resultado,
            status_code=200,
            status_message="Prediction successful"
        )

        return resultado, 200

    except ValidationError as exc:
        registrar_predict(
            payload=payload,
            indicadores=indicadores,
            resultado=resultado,
            status_code=400,
            status_message=str(exc)
        )

        raise

    except LookupError as exc:
        registrar_predict(
            payload=payload,
            indicadores=indicadores,
            resultado=resultado,
            status_code=404,
            status_message=str(exc)
        )

        raise

    except Exception:
        registrar_predict(
            payload=payload,
            indicadores=indicadores,
            resultado=resultado,
            status_code=500,
            status_message="Internal server error"
        )

        raise
    