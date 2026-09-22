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

from src.api.predict_log import registrar_predict


# --------------------------------------------------------------------------------------- #
# Instâncias
# --------------------------------------------------------------------------------------- #

api_bp = Blueprint("api", __name__)

DATABASE_URL = os.getenv("DATABASE_URL")
BUCKET = os.getenv("NEON_S3_BUCKET", "model-artifacts")

MODEL_KEY = os.getenv(
    "MODEL_OBJECT_KEY",
    "models/lints_bundle.joblib"
)

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

_bundle = None


# --------------------------------------------------------------------------------------- #
# Funções auxiiares
# --------------------------------------------------------------------------------------- #

def get_connection():

    return psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row
    )


def get_bundle():

    global _bundle

    if _bundle is None:

        response = s3.get_object(
            Bucket=BUCKET,
            Key=MODEL_KEY
        )

        _bundle = joblib.load(
            io.BytesIO(response["Body"].read())
        )

    return _bundle


def consultar_indicadores(month, year, month_position):

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

                indicadores[indicador] = float(
                    row[indicador]
                )

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

            indicadores["euribor3m"] = float(
                row["euribor3m"]
            )

    return indicadores


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
        raise ValueError(
            f"Campos obrigatórios ausentes: {', '.join(missing)}."
        )

    try:
        age = int(payload["age"])
    except (TypeError, ValueError):
        raise ValueError(
            "age deve ser um número inteiro."
        )

    if not 17 <= age <= 98:
        raise ValueError(
            "age deve estar entre 17 e 98."
        )

    try:
        previous = int(payload["previous"])
    except (TypeError, ValueError):
        raise ValueError(
            "previous deve ser um número inteiro."
        )

    if payload["poutcome"] == "nonexistent":

        if previous != 0:
            raise ValueError(
                "Quando poutcome='nonexistent', previous deve ser 0."
            )

    elif not 1 <= previous <= 7:

        raise ValueError(
            "Quando poutcome não é 'nonexistent', previous deve estar entre 1 e 7."
        )

    return age, previous


def obter_categorias_modelo():

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
            value.item()
            if hasattr(value, "item")
            else value
            for value in values
        ]
        for feature, values
        in zip(
            categorical_features,
            encoder.categories_
        )
    }

    if "day_of_week" in categories:
        categories["day_of_week"] = categories.pop(
            "day_of_week"
        )

    categories["month_position"] = [
        "início",
        "meio",
        "fim"
    ]

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
                row["year"]
                for row in cur.fetchall()
            ]

    return categories

# --------------------------------------------------------------------------------------- #
# Endpoint GET categories
# --------------------------------------------------------------------------------------- #

@api_bp.get("/categories")
def categories():
    """
    Retorna as categorias aceitas pela API.
    ---
    tags:
      - Model
    responses:
      200:
        description: Categorias disponíveis para as variáveis de entrada.
      500:
        description: Erro ao carregar categorias.
    """

    try:

        return jsonify(
            categories=obter_categorias_modelo()
        ), 200

    except Exception as exc:

        return jsonify(
            error=str(exc)
        ), 500

# --------------------------------------------------------------------------------------- #
# Endpoint POST predict
# --------------------------------------------------------------------------------------- #

@api_bp.post("/predict")
def predict():
    """
    Recomenda o canal de contato e estima a conversão.
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
              example: 05. may
            day_of_week:
              type: string
              example: 1. mon
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
        description: Indicadores não encontrados.
      500:
        description: Erro interno.
    """

    payload = request.get_json(
        silent=True
    ) or {}

    try:

        age, previous = validar_payload(
            payload
        )

        categorias = obter_categorias_modelo()

        for field in [
            "job",
            "marital",
            "education",
            "housing",
            "loan",
            "month",
            "day_of_week",
            "poutcome",
            "month_position",
            "year"
        ]:

            if payload[field] not in categorias[field]:
                raise ValueError(
                    f"Valor inválido para {field}. Valores aceitos: {categorias[field]}."
                )

        indicadores = consultar_indicadores(
            payload["month"],
            int(payload["year"]),
            payload["month_position"]
        )

        bundle = get_bundle()

        preprocessor = bundle["preprocessor"]
        lints = bundle["bandit"]
        reward_models = bundle["reward_models"]
        context_features = bundle["context_features"]

        input_model = {
            "age": age,
            "job": payload["job"],
            "marital": payload["marital"],
            "education": payload["education"],
            "housing": payload["housing"],
            "loan": payload["loan"],
            "month": payload["month"],
            "day_of_week": payload["day_of_week"],
            "previous": previous,
            "poutcome": payload["poutcome"],
            "emp_var_rate": indicadores["emp_var_rate"],
            "cons_price_idx": indicadores["cons_price_idx"],
            "cons_conf_idx": indicadores["cons_conf_idx"],
            "euribor3m": indicadores["euribor3m"],
            "nr_employed": indicadores["nr_employed"]
        }

        import pandas as pd

        X = pd.DataFrame(
            [input_model]
        )[context_features]

        X_transformed = np.asarray(
            preprocessor.transform(X),
            dtype=np.float64
        )

        recommended_contact = np.asarray(
            lints.predict(
                contexts=X_transformed
            )
        ).reshape(-1)[0]

        probability = float(
            reward_models[
                recommended_contact
            ].predict_proba(
                X_transformed
            )[0, 1]
        )

        accepted = probability >= 0.5

        result = {
            "recommended_contact": str(
                recommended_contact
            ),
            "conversion_probability": probability,
            "accept_offer": bool(accepted),
            "threshold": 0.5
        }

        registrar_predict(
            payload=payload,
            indicadores=indicadores,
            resultado=result,
            status_code=200,
            status_message="Prediction successful"
        )

        return jsonify(result), 200

    except ValueError as exc:

        registrar_predict(
            payload=payload,
            indicadores={},
            resultado={},
            status_code=400,
            status_message=str(exc)
        )

        return jsonify(
            error=str(exc)
        ), 400

    except LookupError as exc:

        registrar_predict(
            payload=payload,
            indicadores={},
            resultado={},
            status_code=404,
            status_message=str(exc)
        )

        return jsonify(
            error=str(exc)
        ), 404

    except Exception as exc:

        registrar_predict(
            payload=payload,
            indicadores={},
            resultado={},
            status_code=500,
            status_message="Internal server error"
        )

        return jsonify(
            error="Internal server error"
        ), 500

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
        description: Erro ao recuperar as métricas.
    """

    try:
        bundle = get_bundle()

        lints_object = s3.get_object(Bucket=BUCKET, Key="models/results/metricas_lints.csv")
        reward_object = s3.get_object(Bucket=BUCKET, Key="models/results/metricas_reward_models.csv")

        lints_df = pd.read_csv(io.BytesIO(lints_object["Body"].read()))
        reward_df = pd.read_csv(io.BytesIO(reward_object["Body"].read()))

        metadata = bundle.get("metadata", {})

        reward_parameters = {}

        for arm, model in bundle["reward_models"].items():
            reward_parameters[str(arm)] = {
                key: value
                for key, value in model.get_params().items()
                if isinstance(value, (str, int, float, bool, type(None)))
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
                "metrics": lints_df.to_dict(orient="records")
            },
            "reward_models": {
                "model": "LogisticRegression",
                "parameters": reward_parameters,
                "metrics": reward_metrics
            }
        }), 200

    except Exception as exc:
        return jsonify(error=str(exc)), 500

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
        "neon_object_storage": "error",
        "model": "error",
        "indicators": "error"
    }

    try:

        with get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    "SELECT 1 AS health"
                )

                if cur.fetchone()["health"] == 1:
                    checks[
                        "neon_postgres"
                    ] = "ok"

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
                    checks["indicators"] = "ok"

    except Exception:
        pass

    try:

        s3.head_object(
            Bucket=BUCKET,
            Key=MODEL_KEY
        )

        checks[
            "neon_object_storage"
        ] = "ok"

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

        if required.issubset(
            bundle.keys()
        ):
            checks["model"] = "ok"

    except Exception:
        pass

    healthy = all(
        value == "ok"
        for value in checks.values()
    )

    return jsonify({
        "status": (
            "healthy"
            if healthy
            else "unhealthy"
        ),
        "checks": checks
    }), 200 if healthy else 503
