"""Aplicação Flask - API de recomendação de ofertas (multi-armed bandit).

Execução:
    python3 main.py
"""

import os

from flask import Flask, jsonify, render_template, request

from src.api import home_backend

app = Flask(
    __name__,
    template_folder="src/templates",
    static_folder="src/static",
)

GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/")
SWAGGER_URL = os.getenv("SWAGGER_URL", "/docs")


@app.get("/")
def home():
    return render_template("home.html", github_url=GITHUB_URL, swagger_url=SWAGGER_URL)


@app.get("/home/indicators")
def home_indicators():
    """Indicadores econômicos para o mês/ano (e período do mês) informados."""
    try:
        indicators = home_backend.get_indicators(
            request.args.get("month"),
            request.args.get("year"),
            request.args.get("periodo_mes"),
        )
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 500
    return jsonify(indicators=indicators)


@app.post("/home/predict")
def home_predict():
    """Contato recomendado e decisão de aceitação da oferta."""
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(home_backend.predict(payload))
    except home_backend.ValidationError as exc:
        return jsonify(error=str(exc)), 400
    except LookupError as exc:
        return jsonify(error=str(exc)), 404
    except (FileNotFoundError, KeyError) as exc:
        return jsonify(error=str(exc)), 500


if __name__ == "__main__":
    # Porta 5000 costuma estar ocupada pelo AirPlay Receiver no macOS
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=True)
