"""Adaptador entre a interface home e a lógica central da API."""

from __future__ import annotations

from typing import Any

from src.api.api_endpoints import (
    ValidationError,
    indicadores_para_home
)
from src.api.predict_log import processar_predict


CONTACT_LABELS = {
    "cellular": "Celular",
    "telephone": "Telefone"
}


def get_indicators(month: str | None, year: str | int | None, periodo_mes: str | None = None) -> list[dict[str, Any]]:
    """Retorna os indicadores no formato esperado pela página home."""

    return indicadores_para_home(
        month=month,
        year=year,
        month_position=periodo_mes
    )


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Executa a predição, registra a requisição e adapta o resultado para a home."""

    api_payload = {
        "age": payload.get("age"),
        "job": payload.get("job"),
        "marital": payload.get("marital"),
        "education": payload.get("education"),
        "housing": payload.get("housing"),
        "loan": payload.get("loan"),
        "month": payload.get("month"),
        "day_of_week": payload.get("day_of_week"),
        "poutcome": payload.get("poutcome"),
        "month_position": payload.get(
            "month_position",
            payload.get("periodo_mes")
        ),
        "year": payload.get("year"),
        "previous": payload.get("previous")
    }

    result, _ = processar_predict(api_payload)

    contact = result["recommended_contact"]
    probability = float(result["conversion_probability"])
    accepts = bool(result["accept_offer"])

    return {
        "contact_recomendado": contact,
        "contact_label": CONTACT_LABELS.get(contact.lower(), contact),
        "probabilidade_conversao": probability,
        "aceita_oferta": "yes" if accepts else "no",
        "aceita_label": "Sim" if accepts else "Não",
        "threshold": float(result["threshold"])
    }
