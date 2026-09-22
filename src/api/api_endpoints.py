# --------------------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------------------- #

import io
import os

import boto3
import joblib
import numpy as np
import pandas as pd
import psycopg
from flask import Blueprint, jsonify, request
from psycopg.rows import dict_row

from src.api.predict_log import processar_predict

from src.config.neon import (
    DATABASE_URL,
    NEON_S3_BUCKET,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_ENDPOINT_URL_S3,
    AWS_REGION,
    validar_database,
    validar_object_storage
)

# --------------------------------------------------------------------------------------- #
# Instâncias
# --------------------------------------------------------------------------------------- #

api_bp = Blueprint("api", __name__)

BUCKET = NEON_S3_BUCKET

MODEL_KEY = os.getenv("MODEL_OBJECT_KEY", "models/lints_bundle.joblib")
LINTS_METRICS_KEY = os.getenv("LINTS_METRICS_KEY", "models/results/metricas_lints.csv")
REWARD_METRICS_KEY = os.getenv("REWARD_METRICS_KEY", "models/results/metricas_reward_models.csv")

THRESHOLD = 0.5

INDICATOR_LABELS = {
    "emp_var_rate": "Taxa de variação de emprego",
    "cons_price_idx": "Índice de preço ao consumidor",
    "cons_conf_idx": "Índice de confiança do consumidor",
    "euribor3m": "Euribor 3m",
    "nr_employed": "Número de empregados"
}

MONTH_POSITION_OPTIONS = ["início", "meio", "fim"]

_bundle = None


class ValidationError(ValueError):
    pass


# --------------------------------------------------------------------------------------- #
# Funções auxiiares
# --------------------------------------------------------------------------------------- #

def get_connection():

    validar_database()

    return psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row
    )


def get_s3_client():

    validar_object_storage()

    return boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL_S3,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


def get_bundle():

    global _bundle

    if _bundle is None:

        s3 = get_s3_client()

        response = s3.get_object(
            Bucket=BUCKET,
            Key=MODEL_KEY
        )

        _bundle = joblib.load(
            io.BytesIO(response["Body"].read())
        )

    return _bundle


def normalizar_month_position(value):

    if value is None:
        return None

    value = str(value).strip().lower()

    mapping = {
        "início": "início",
        "meio": "meio",
        "fim": "fim"
    }

    return mapping.get(value, value)


def consultar_indicadores(month, year, month_position):

    if month in (None, ""):
        raise ValidationError("month é obrigatório.")

    try:
        year = int(year)
    except (TypeError, ValueError):
        raise ValidationError("year deve ser um número inteiro.") from None

    month_position = normalizar_month_position(month_position)

    if month_position not in ["início", "meio", "fim"]:
        raise ValidationError("month_position deve ser 'início', 'meio' ou 'fim'.")

    tabelas = {
        "emp_var_rate": "emp_var_rate_mensal",
        "cons_price_idx": "cons_price_idx_mensal",
        "cons_conf_idx": "cons_conf_idx_mensal",
        "nr_employed": "nr_employed_mensal"
    }

    indicadores = {}

    with get_connection() as conn:

        with conn.cursor() as cur:

            for indicador, tabela in tabelas.items():

                cur.execute(
                    f"""
                    SELECT {indicador}
                    FROM {tabela}
                    WHERE month = %s
                    AND year = %s
                    """,
                    (month, year)
                )

                row = cur.fetchone()

                if row is None:
                    raise LookupError(
                        f"Não existem dados de {indicador} para month={month} e year={year}."
                    )

                indicadores[indicador] = float(row[indicador])

            cur.execute(
                """
                SELECT euribor3m
                FROM euribor3m_mensal
                WHERE month = %s
                AND year = %s
                AND month_position = %s
                """,
                (month, year, month_position)
            )

            row = cur.fetchone()

            if row is None:
                raise LookupError(
                    f"Não existem dados de euribor3m para month={month}, year={year} e month_position={month_position}."
                )

            indicadores["euribor3m"] = float(row["euribor3m"])

    return indicadores


def indicadores_para_home(month, year, month_position):

    indicadores = consultar_indicadores(
        month,
        year,
        month_position
    )

    return [
        {
            "key": key,
            "label": INDICATOR_LABELS[key],
            "value": indicadores[key]
        }
        for key in [
            "emp_var_rate",
            "cons_price_idx",
            "cons_conf_idx",
            "euribor3m",
            "nr_employed"
        ]
    ]


def obter_categorias():

    bundle = get_bundle()

    preprocessor = bundle["preprocessor"]

    categorical_features = bundle.get(
        "categorical_features",
        [
            "job",
            "marital",
            "education",
            "housing",
            "loan",
            "month",
            "day_of_week",
            "poutcome"
        ]
    )

    encoder = preprocessor.named_transformers_["cat"]

    categories = {
        feature: [
            value.item() if hasattr(value, "item") else value
            for value in values
        ]
        for feature, values in zip(categorical_features, encoder.categories_)
    }

    if "day_of_week" in categories:
        categories["day_of_week"] = categories.pop("day_of_week")

    categories["month_position"] = MONTH_POSITION_OPTIONS

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT DISTINCT year
                FROM emp_var_rate_mensal
                ORDER BY year
                """
            )

            categories["year"] = [
                int(row["year"])
                for row in cur.fetchall()
            ]

    return categories


def validar_payload(payload):

    required = [
        "age",
        "job",
        "marital",
        "education",
        "housing",
        "loan",
        "month",
        "day_of_week",
        "poutcome",
        "month_position",
        "year",
        "previous"
    ]

    missing = [
        field
        for field in required
        if field not in payload
    ]

    if missing:
        raise ValidationError(
            f"Campos obrigatórios ausentes: {', '.join(missing)}."
        )

    try:
        age = int(payload["age"])
    except (TypeError, ValueError):
        raise ValidationError("age deve ser um número inteiro.") from None

    if not 17 <= age <= 98:
        raise ValidationError("age deve estar entre 17 e 98.")

    try:
        year = int(payload["year"])
    except (TypeError, ValueError):
        raise ValidationError("year deve ser um número inteiro.") from None

    try:
        previous = int(payload["previous"])
    except (TypeError, ValueError):
        raise ValidationError("previous deve ser um número inteiro.") from None

    if payload["poutcome"] == "nonexistent":

        if previous != 0:
            raise ValidationError(
                "Quando poutcome='nonexistent', previous deve ser 0."
            )

    elif not 1 <= previous <= 7:

        raise ValidationError(
            "Quando poutcome não é 'nonexistent', previous deve estar entre 1 e 7."
        )

    categories = obter_categorias()

    fields_to_validate = [
        "job",
        "marital",
        "education",
        "housing",
        "loan",
        "month",
        "day_of_week",
        "poutcome"
    ]

    for field in fields_to_validate:

        if payload[field] not in categories[field]:
            raise ValidationError(
                f"Valor inválido para {field}: {payload[field]!r}."
            )

    if payload["month_position"] not in MONTH_POSITION_OPTIONS:
        raise ValidationError(
            "month_position deve ser 'início', 'meio' ou 'fim'."
        )

    if year not in categories["year"]:
        raise ValidationError(
            f"year inválido. Valores aceitos: {categories['year']}."
        )

    return {
        **payload,
        "age": age,
        "year": year,
        "previous": previous
    }


def executar_predict(payload):

    clean = validar_payload(payload)

    indicadores = consultar_indicadores(
        clean["month"],
        clean["year"],
        clean["month_position"]
    )

    bundle = get_bundle()

    preprocessor = bundle["preprocessor"]
    lints = bundle["bandit"]
    reward_models = bundle["reward_models"]
    context_features = list(bundle["context_features"])

    input_model = {
        "age": clean["age"],
        "job": clean["job"],
        "marital": clean["marital"],
        "education": clean["education"],
        "housing": clean["housing"],
        "loan": clean["loan"],
        "month": clean["month"],
        "day_of_week": clean["day_of_week"],
        "previous": clean["previous"],
        "poutcome": clean["poutcome"],
        "emp_var_rate": indicadores["emp_var_rate"],
        "cons_price_idx": indicadores["cons_price_idx"],
        "cons_conf_idx": indicadores["cons_conf_idx"],
        "euribor3m": indicadores["euribor3m"],
        "nr_employed": indicadores["nr_employed"]
    }

    df = pd.DataFrame(
        [{feature: input_model[feature] for feature in context_features}]
    )[context_features]

    X = np.asarray(
        preprocessor.transform(df),
        dtype=np.float64
    )

    action = np.asarray(
        lints.predict(contexts=X)
    ).reshape(-1)[0]

    if action not in reward_models:
        action = next(
            arm
            for arm in reward_models
            if str(arm) == str(action)
        )

    probability = float(
        reward_models[action].predict_proba(X)[0, 1]
    )

    result = {
        "recommended_contact": str(action),
        "conversion_probability": probability,
        "accept_offer": bool(probability >= THRESHOLD),
        "threshold": THRESHOLD
    }

    return result, indicadores

# --------------------------------------------------------------------------------------- #
# Endpoint GET categories
# --------------------------------------------------------------------------------------- #

@api_bp.get("/categories")
def categories():
    """
    Retorna todas as categorias aceitas pela API.
    ---
    tags:
      - Model
    responses:
      200:
        description: Categorias disponíveis.
      500:
        description: Erro interno.
    """

    try:
        return jsonify(
            categories=obter_categorias()
        ), 200

    except Exception:
        return jsonify(
            error="Erro interno ao carregar categorias."
        ), 500

# --------------------------------------------------------------------------------------- #
# Endpoint GET indicators
# --------------------------------------------------------------------------------------- #

@api_bp.get("/indicators")
def indicators():
    """
    Retorna os indicadores econômicos do período informado.
    ---
    tags:
      - Indicators
    parameters:
      - name: month
        in: query
        type: string
        required: true
      - name: year
        in: query
        type: integer
        required: true
      - name: month_position
        in: query
        type: string
        required: true
        enum: [início, meio, fim]
    responses:
      200:
        description: Indicadores encontrados.
      400:
        description: Entrada inválida.
      404:
        description: Combinação de período inexistente.
      500:
        description: Erro interno.
    """

    try:
        result = indicadores_para_home(
            request.args.get("month"),
            request.args.get("year"),
            request.args.get("month_position")
        )

        return jsonify(
            indicators=result
        ), 200

    except ValidationError as exc:
        return jsonify(error=str(exc)), 400

    except LookupError as exc:
        return jsonify(error=str(exc)), 404

    except Exception:
        return jsonify(error="Erro interno ao consultar indicadores."), 500

# --------------------------------------------------------------------------------------- #
# Endpoint POST predict
# --------------------------------------------------------------------------------------- #

@api_bp.post("/predict")
def predict():
    """
    Recomenda o canal de contato e estima a probabilidade de conversão.
    ---
    tags:
      - Model
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - age
            - job
            - marital
            - education
            - housing
            - loan
            - month
            - day_of_week
            - poutcome
            - month_position
            - year
            - previous
          properties:
            age:
              type: integer
              example: 35
            job:
              type: string
              example: admin.
            marital:
              type: string
              example: married
            education:
              type: string
              example: university.degree
            housing:
              type: string
              example: unknown
            loan:
              type: string
              example: unknown
            month:
              type: string
              example: "05. may"
            day_of_week:
              type: string
              example: "1. mon"
            poutcome:
              type: string
              example: nonexistent
            month_position:
              type: string
              enum: [início, meio, fim]
              example: meio
            year:
              type: integer
              example: 2009
            previous:
              type: integer
              example: 0
    responses:
      200:
        description: Predição realizada com sucesso.
      400:
        description: Entrada inválida.
      404:
        description: Indicadores não encontrados para o período.
      500:
        description: Erro interno.
    """

    payload = request.get_json(silent=True) or {}

    try:
        resultado, status_code = processar_predict(payload)

        return jsonify(resultado), status_code

    except ValidationError as exc:
        return jsonify(error=str(exc)), 400

    except LookupError as exc:
        return jsonify(error=str(exc)), 404

    except Exception:
        return jsonify(error="Internal server error"), 500

# --------------------------------------------------------------------------------------- #
# Endpoint GET metrics
# --------------------------------------------------------------------------------------- #

@api_bp.get("/metrics")
def metrics():
    """
    Retorna parâmetros e métricas dos modelos.
    ---
    tags:
      - Model
    responses:
      200:
        description: Métricas do LinTS e dos reward models.
      500:
        description: Erro ao recuperar métricas.
    """

    try:

        s3 = get_s3_client()

        lints_object = s3.get_object(
            Bucket=BUCKET,
            Key=LINTS_METRICS_KEY
        )

        reward_object = s3.get_object(
            Bucket=BUCKET,
            Key=REWARD_METRICS_KEY
        )

        lints_df = pd.read_csv(
            io.BytesIO(
                lints_object["Body"].read()
            )
        )

        reward_df = pd.read_csv(
            io.BytesIO(
                reward_object["Body"].read()
            )
        )

        bundle = get_bundle()

        metadata = bundle.get(
            "metadata",
            {}
        )

        reward_parameters = {}

        for arm, model in bundle["reward_models"].items():

            reward_parameters[str(arm)] = {
                key: value
                for key, value in model.get_params().items()
                if isinstance(
                    value,
                    (str, int, float, bool, type(None))
                )
            }

        reward_metrics = {}

        for _, row in reward_df.iterrows():

            arm = str(row["arm"])

            reward_metrics[arm] = {
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"])
            }

        return jsonify({
            "lints": {
                "parameters": {
                    "algorithm": "Linear Thompson Sampling",
                    "alpha": metadata.get("alpha"),
                    "l2_lambda": metadata.get("l2_lambda"),
                    "seed": metadata.get("seed"),
                    "arms": metadata.get("arms")
                },
                "metrics": lints_df.to_dict(
                    orient="records"
                )
            },
            "reward_models": {
                "model": "LogisticRegression",
                "parameters": reward_parameters,
                "metrics": reward_metrics
            }
        }), 200

    except Exception:
        return jsonify(
            error="Erro interno ao recuperar métricas."
        ), 500

# --------------------------------------------------------------------------------------- #
# Endpoint GET health
# --------------------------------------------------------------------------------------- #

@api_bp.get("/health")
def health():
    """
    Verifica a saúde da API e suas dependências.
    ---
    tags:
      - Health
    responses:
      200:
        description: API e dependências saudáveis.
      503:
        description: Uma ou mais dependências estão indisponíveis.
    """

    checks = {
        "api": "ok",
        "neon_postgres": "error",
        "indicator_tables": "error",
        "neon_object_storage": "error",
        "model": "error"
    }

    try:

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    "SELECT 1 AS health"
                )

                if cur.fetchone()["health"] == 1:
                    checks["neon_postgres"] = "ok"

                cur.execute(
                    """
                    SELECT
                        to_regclass('public.emp_var_rate_mensal') IS NOT NULL
                        AND to_regclass('public.cons_price_idx_mensal') IS NOT NULL
                        AND to_regclass('public.cons_conf_idx_mensal') IS NOT NULL
                        AND to_regclass('public.euribor3m_mensal') IS NOT NULL
                        AND to_regclass('public.nr_employed_mensal') IS NOT NULL
                        AS healthy
                    """
                )

                if cur.fetchone()["healthy"]:
                    checks["indicator_tables"] = "ok"

    except Exception:
        pass

    try:

        s3 = get_s3_client()

        s3.head_object(
            Bucket=BUCKET,
            Key=MODEL_KEY
        )

        checks["neon_object_storage"] = "ok"

    except Exception:
        pass

    try:

        bundle = get_bundle()

        required = {
            "preprocessor",
            "bandit",
            "reward_models",
            "context_features"
        }

        if required.issubset(bundle.keys()):
            checks["model"] = "ok"

    except Exception:
        pass

    healthy = all(
        value == "ok"
        for value in checks.values()
    )

    return jsonify({
        "status": "healthy" if healthy else "unhealthy",
        "checks": checks
    }), 200 if healthy else 503
