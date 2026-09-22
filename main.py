# --------------------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------------------- #

import os

from dotenv import load_dotenv
from flasgger import Swagger
from flask import Flask, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

from src.api import home_backend
from src.api.api_endpoints import api_bp


load_dotenv()

# --------------------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------------------- #

def create_app():

    app = Flask(
        __name__,
        template_folder="src/templates",
        static_folder="src/static"
    )

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    app.config["SWAGGER"] = {
        "title": "Financial ML Decisioning API",
        "description": "API de recomendação de canal utilizando Linear Thompson Sampling.",
        "version": "1.0.0",
        "uiversion": 3,
        "specs_route": "/docs/"
    }

    Swagger(app)

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def home():

        return render_template(
            "home.html",
            github_url=os.getenv("GITHUB_URL", "https://github.com/"),
            swagger_url="/docs/"
        )

    @app.get("/home/indicators")
    def home_indicators():
        """Rota de compatibilidade utilizada pela interface home."""

        try:
            indicators = home_backend.get_indicators(
                request.args.get("month"),
                request.args.get("year"),
                request.args.get("periodo_mes")
            )

            return jsonify(indicators=indicators), 200

        except home_backend.ValidationError as exc:
            return jsonify(error=str(exc)), 400

        except LookupError as exc:
            return jsonify(error=str(exc)), 404

        except Exception:
            return jsonify(error="Erro interno ao consultar indicadores."), 500

    @app.post("/home/predict")
    def home_predict():
        """Rota de compatibilidade utilizada pela interface home."""

        payload = request.get_json(silent=True) or {}

        try:
            return jsonify(home_backend.predict(payload)), 200

        except home_backend.ValidationError as exc:
            return jsonify(error=str(exc)), 400

        except LookupError as exc:
            return jsonify(error=str(exc)), 404

        except Exception:
            return jsonify(error="Erro interno ao realizar a predição."), 500

    return app


app = create_app()


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )
