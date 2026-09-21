"""Lógica de backend da página home.

- Consulta dos indicadores econômicos nas tabelas parquet.
- Inferência do modelo LinTS (multi-armed bandit) + modelos de recompensa.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

# --------------------------------------------------------------------------- #
# Configuração (pode ser sobrescrita por variáveis de ambiente)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # raiz do repositório (onde está o main.py)
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "trusted"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "lints_bundle.joblib"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.30"))

# Indicadores: (chave, rótulo exibido, arquivo parquet, usa periodo_mes?)
INDICATORS = [
    ("emp_var_rate", "Taxa de variação de emprego", "emp_var_rate_mensal.parquet", False),
    ("cons_price_idx", "Índice de preço ao consumidor", "cons_price_idx_mensal.parquet", False),
    ("cons_conf_idx", "Índice de confiança do consumidor", "cons_conf_idx_mensal.parquet", False),
    ("euribor3m", "Euribor 3m", "euribor3m_mensal.parquet", True),
    ("nr_employed", "Número de empregados", "nr_employed_mensal.parquet", False),
]

# Categorias válidas (validação server-side)
CATEGORIES: dict[str, list[str]] = {
    "job": ["student", "housemaid", "services", "admin.", "blue-collar", "technician",
            "management", "unemployed", "self-employed", "entrepreneur", "retired", "unknown"],
    "marital": ["single", "married", "divorced", "unknown"],
    "education": ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",
                  "professional.course", "university.degree", "unknown"],
    "loan": ["yes", "no", "unknown"],
    "housing": ["yes", "no", "unknown"],
    "month": ["03. mar", "04. apr", "05. may", "06. jun", "07. jul",
              "08. aug", "09. sep", "10. oct", "11. nov", "12. dec"],
    "periodo_mes": ["início", "meio", "fim"],
    "day_of_week": ["1. mon", "2. tue", "3. wed", "4. thu", "5. fri"],
    "poutcome": ["success", "failure", "nonexistent"],
}
VALID_YEARS = [2008, 2009, 2010]
AGE_MIN, AGE_MAX = 17, 98

# Valor de "periodo_mes" como o modelo/parquet espera (ajuste se necessário)
PERIODO_MES_MODEL_VALUE = {"início": "início", "meio": "meio", "fim": "fim"}

# Nome da coluna do modelo para cada campo do formulário/indicador.
# Ajuste aqui se as colunas de `context_features` tiverem outros nomes.
FEATURE_COLUMNS = {
    "age": "age",
    "job": "job",
    "marital": "marital",
    "education": "education",
    "loan": "loan",
    "housing": "housing",
    "month": "month",
    "year": "year",
    "periodo_mes": "periodo_mes",
    "day_of_week": "day_of_week",
    "poutcome": "poutcome",
    "emp_var_rate": "emp_var_rate",
    "cons_price_idx": "cons_price_idx",
    "cons_conf_idx": "cons_conf_idx",
    "euribor3m": "euribor3m",
    "nr_employed": "nr_employed",
}

CONTACT_LABELS = {"cellular": "Celular", "telephone": "Telefone"}


class ValidationError(ValueError):
    """Erro de validação de entrada (HTTP 400)."""


# --------------------------------------------------------------------------- #
# Normalização de chaves (mês / ano / período do mês)
# --------------------------------------------------------------------------- #
_MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_PERIODO_ALIASES = {
    "inicio": "inicio", "begin": "inicio", "beginning": "inicio", "start": "inicio",
    "meio": "meio", "mid": "meio", "middle": "meio",
    "fim": "fim", "end": "fim",
}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _month_number(value: Any) -> int | None:
    """Aceita 3, '3', '03. mar', 'mar', 'March'... e devolve 1-12."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value)
    text = str(value).strip().lower()
    match = re.match(r"^(\d{1,2})", text)
    if match:
        return int(match.group(1))
    return _MONTH_NAMES.get(text[:3])


def _periodo_key(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = _strip_accents(str(value).strip().lower())
    text = re.sub(r"^\d+\.\s*", "", text)  # remove prefixo numérico, ex.: "1. inicio"
    return _PERIODO_ALIASES.get(text, text)


# --------------------------------------------------------------------------- #
# Indicadores econômicos
# --------------------------------------------------------------------------- #
_tables: dict[str, pd.DataFrame] = {}
_tables_lock = threading.Lock()


def _load_table(filename: str) -> pd.DataFrame:
    with _tables_lock:
        if filename not in _tables:
            path = DATA_DIR / filename
            if not path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {path}")
            df = pd.read_parquet(path)
            df["_month"] = df["month"].map(_month_number)
            df["_year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            if "periodo_mes" in df.columns:
                df["_periodo"] = df["periodo_mes"].map(_periodo_key)
            _tables[filename] = df
        return _tables[filename]


def _value_column(df: pd.DataFrame) -> str:
    ignored = {"month", "year", "periodo_mes"}
    candidates = [c for c in df.columns if c not in ignored and not c.startswith("_")]
    if not candidates:
        raise ValueError("Não foi possível identificar a coluna de valor na tabela parquet.")
    return candidates[0]


def get_indicators(month: str | None, year: str | int | None,
                   periodo_mes: str | None = None) -> list[dict[str, Any]]:
    """Retorna a lista de indicadores para o período informado.

    Os 4 indicadores mensais dependem de mês e ano; o Euribor 3m depende também
    do período do mês. Quando não há dado, o valor vem como None.
    """
    month_num = _month_number(month)
    try:
        year_num = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year_num = None
    periodo = _periodo_key(periodo_mes)

    result = []
    for key, label, filename, needs_periodo in INDICATORS:
        value = None
        if month_num is not None and year_num is not None and (not needs_periodo or periodo):
            df = _load_table(filename)
            mask = (df["_month"] == month_num) & (df["_year"] == year_num)
            if needs_periodo:
                mask &= df["_periodo"] == periodo
            rows = df.loc[mask]
            if not rows.empty:
                value = float(rows.iloc[0][_value_column(df)])
        result.append({"key": key, "label": label, "value": value})
    return result


# --------------------------------------------------------------------------- #
# Modelo (LinTS + modelos de recompensa)
# --------------------------------------------------------------------------- #
_bundle: dict[str, Any] | None = None
_bundle_lock = threading.Lock()


def _get_bundle() -> dict[str, Any]:
    global _bundle
    with _bundle_lock:
        if _bundle is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
            _bundle = load(MODEL_PATH)
        return _bundle


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}

    try:
        age = int(str(payload.get("age", "")).strip())
    except ValueError:
        raise ValidationError("idade: informe apenas números") from None
    if not AGE_MIN <= age <= AGE_MAX:
        raise ValidationError(f"idade: valor deve estar entre {AGE_MIN} e {AGE_MAX}")
    clean["age"] = age

    try:
        year = int(payload.get("year"))
    except (TypeError, ValueError):
        raise ValidationError("ano inválido") from None
    if year not in VALID_YEARS:
        raise ValidationError("ano inválido")
    clean["year"] = year

    for field, allowed in CATEGORIES.items():
        value = payload.get(field)
        if value not in allowed:
            raise ValidationError(f"Valor inválido para '{field}': {value!r}")
        clean[field] = value
    return clean


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Executa o modelo para um único cliente e devolve contato + aceitação."""
    data = _validate(payload)

    # Indicadores vêm do servidor (não confiamos nos valores do cliente)
    indicators = get_indicators(data["month"], data["year"], data["periodo_mes"])
    for item in indicators:
        if item["value"] is None:
            raise LookupError(f"Indicador sem dados para o período: {item['label']}")
        data[item["key"]] = item["value"]

    data["periodo_mes"] = PERIODO_MES_MODEL_VALUE.get(data["periodo_mes"], data["periodo_mes"])

    bundle = _get_bundle()
    preprocessor = bundle["preprocessor"]
    lints = bundle["bandit"]
    reward_models = bundle["reward_models"]
    context_features = bundle["context_features"]

    row = {FEATURE_COLUMNS[k]: v for k, v in data.items() if k in FEATURE_COLUMNS}
    missing = [c for c in context_features if c not in row]
    if missing:
        raise KeyError(
            f"O modelo espera colunas que a home não fornece: {missing}. "
            "Ajuste FEATURE_COLUMNS em home_backend.py."
        )

    df = pd.DataFrame([row])[list(context_features)]
    X = np.asarray(preprocessor.transform(df), dtype=np.float64)

    action = np.asarray(lints.predict(contexts=X)).reshape(-1)[0]
    if action not in reward_models:  # tolera diferença de tipo (ex.: np.str_ vs str)
        action = next(a for a in reward_models if str(a) == str(action))
    probability = float(reward_models[action].predict_proba(X)[:, 1][0])

    accepts = "yes" if probability >= THRESHOLD else "no"
    contact = str(action)
    return {
        "contact_recomendado": contact,
        "contact_label": CONTACT_LABELS.get(contact.lower(), contact),
        "probabilidade_conversao": probability,
        "aceita_oferta": accepts,
        "aceita_label": "Sim" if accepts == "yes" else "Não",
        "threshold": THRESHOLD,
    }
