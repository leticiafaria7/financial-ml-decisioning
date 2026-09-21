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
THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

# Indicadores: (chave = nome da coluna de valor, rótulo exibido, arquivo parquet, usa month_position?)
INDICATORS = [
    ("emp_var_rate", "Taxa de variação de emprego", "emp_var_rate_mensal.parquet", False),
    ("cons_price_idx", "Índice de preço ao consumidor", "cons_price_idx_mensal.parquet", False),
    ("cons_conf_idx", "Índice de confiança do consumidor", "cons_conf_idx_mensal.parquet", False),
    ("euribor3m", "Euribor 3m", "euribor3m_mensal.parquet", True),
    ("nr_employed", "Número de empregados", "nr_employed_mensal.parquet", False),
]

# Colunas que identificam a linha nas tabelas (não são o valor do indicador)
KEY_COLUMNS = {"month", "year", "month_position", "periodo_mes"}
PERIOD_COLUMN_NAMES = ("month_position", "periodo_mes")

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
    "day_of_week": ["1. mon", "2. tue", "3. wed", "4. thu", "5. fri"],
    "poutcome": ["success", "failure", "nonexistent"],
}
# Campos do formulário usados apenas para buscar o Euribor (NÃO são variáveis do modelo)
PERIODO_MES_OPTIONS = ["início", "meio", "fim"]
VALID_YEARS = [2008, 2009, 2010]
AGE_MIN, AGE_MAX = 17, 98

# Variáveis do modelo (o bundle traz a lista oficial em `context_features`)
MODEL_FEATURES = [
    "age", "previous", "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m",
    "nr_employed", "job", "marital", "education", "housing", "loan", "month",
    "day_of_week", "poutcome",
]

CONTACT_LABELS = {"cellular": "Celular", "telephone": "Telefone"}


class ValidationError(ValueError):
    """Erro de validação de entrada (HTTP 400)."""


# --------------------------------------------------------------------------- #
# Normalização de chaves (mês / período do mês)
# --------------------------------------------------------------------------- #
_MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_PERIODO_NUMERIC = {"1": "inicio", "1.0": "inicio", "2": "meio", "2.0": "meio", "3": "fim", "3.0": "fim"}
_PERIODO_PREFIXES = (
    (("ini", "beg", "sta", "ear"), "inicio"),
    (("mei", "mid"), "meio"),
    (("fim", "end", "fin", "lat"), "fim"),
)


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
    """Normaliza 'início'/'inicio'/'begin'/'1. início'/1 -> 'inicio' (idem meio, fim)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = _strip_accents(str(value).strip().lower())
    if text in _PERIODO_NUMERIC:
        return _PERIODO_NUMERIC[text]
    text = re.sub(r"^\d+\.\s+", "", text)  # remove prefixo numérico, ex.: "1. inicio"
    for prefixes, key in _PERIODO_PREFIXES:
        if text.startswith(prefixes):
            return key
    return text


# --------------------------------------------------------------------------- #
# Indicadores econômicos
# --------------------------------------------------------------------------- #
_tables: dict[str, pd.DataFrame] = {}
_tables_lock = threading.Lock()


def _period_column(df: pd.DataFrame) -> str | None:
    for name in PERIOD_COLUMN_NAMES:
        if name in df.columns:
            return name
    return None


def _load_table(filename: str) -> pd.DataFrame:
    with _tables_lock:
        if filename not in _tables:
            path = DATA_DIR / filename
            if not path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {path}")
            df = pd.read_parquet(path)
            df["_month"] = df["month"].map(_month_number)
            df["_year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            period_col = _period_column(df)
            if period_col:
                df["_periodo"] = df[period_col].map(_periodo_key)
            _tables[filename] = df
        return _tables[filename]


def _value_column(df: pd.DataFrame, key: str) -> str:
    if key in df.columns:
        return key
    candidates = [c for c in df.columns if c not in KEY_COLUMNS and not c.startswith("_")]
    if not candidates:
        raise ValueError("Não foi possível identificar a coluna de valor na tabela parquet.")
    return candidates[0]


def get_indicators(month: str | None, year: str | int | None,
                   periodo_mes: str | None = None) -> list[dict[str, Any]]:
    """Retorna a lista de indicadores para o período informado.

    Os 4 indicadores mensais dependem de mês e ano; o Euribor 3m depende também
    de month_position (início/meio/fim). Quando não há dado, o valor vem como None.
    """
    month_num = _month_number(month)
    try:
        year_num = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year_num = None
    periodo = _periodo_key(periodo_mes) if periodo_mes else None

    result = []
    for key, label, filename, needs_periodo in INDICATORS:
        value = None
        if month_num is not None and year_num is not None and (not needs_periodo or periodo):
            df = _load_table(filename)
            mask = (df["_month"] == month_num) & (df["_year"] == year_num)
            if needs_periodo:
                if "_periodo" not in df.columns:
                    raise ValueError(
                        f"A tabela {filename} precisa de uma coluna 'month_position' (ou 'periodo_mes')."
                    )
                mask &= df["_periodo"] == periodo
            rows = df.loc[mask.fillna(False)]
            if not rows.empty:
                value = float(rows.iloc[0][_value_column(df, key)])
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

    periodo = payload.get("periodo_mes")
    if periodo not in PERIODO_MES_OPTIONS:
        raise ValidationError(f"Valor inválido para 'periodo_mes': {periodo!r}")
    clean["periodo_mes"] = periodo

    for field, allowed in CATEGORIES.items():
        value = payload.get(field)
        if value not in allowed:
            raise ValidationError(f"Valor inválido para '{field}': {value!r}")
        clean[field] = value
    return clean


def _derive_previous(poutcome: str) -> int:
    """`previous` (nº de contatos antes desta campanha) não está no formulário.

    Aproximação: sem campanha anterior ('nonexistent') => 0; caso contrário => 1.
    Se quiser o valor real, adicione um campo na home e envie `previous` no payload.
    """
    return 0 if poutcome == "nonexistent" else 1


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Executa o modelo para um único cliente e devolve contato + aceitação."""
    data = _validate(payload)

    # Os indicadores vêm do servidor (não confiamos nos valores do cliente).
    # month + year + month_position servem SÓ para achar o Euribor; não entram no modelo.
    indicators = get_indicators(data["month"], data["year"], data["periodo_mes"])
    for item in indicators:
        if item["value"] is None:
            raise LookupError(f"Indicador sem dados para o período: {item['label']}")
        data[item["key"]] = item["value"]

    if "previous" in payload and str(payload["previous"]).strip() != "":
        data["previous"] = int(payload["previous"])
    else:
        data["previous"] = _derive_previous(data["poutcome"])

    bundle = _get_bundle()
    preprocessor = bundle["preprocessor"]
    lints = bundle["bandit"]
    reward_models = bundle["reward_models"]
    context_features = list(bundle["context_features"])

    missing = [c for c in context_features if c not in data]
    if missing:
        raise KeyError(
            f"O modelo espera colunas que a home não fornece: {missing}. "
            "Ajuste os campos em home_backend.py."
        )

    df = pd.DataFrame([{c: data[c] for c in context_features}])[context_features]
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
