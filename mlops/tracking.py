
# --------------------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------------------- #

from pathlib import Path
import mlflow
from src.config.neon import MLFLOW_TRACKING_URI

# --------------------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------------------- #

def configurar_mlflow():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return MLFLOW_TRACKING_URI

# --------------------------------------------------------------------------------------- #
# Registro do LinTS
# --------------------------------------------------------------------------------------- #

def registrar_lints(params, metrics, artifacts=None):

    configurar_mlflow()

    mlflow.set_experiment("contextual_bandit_lints")

    with mlflow.start_run(run_name="LinTS") as run:

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if artifacts:

            for artifact in artifacts:

                artifact = Path(artifact)

                if artifact.exists():
                    mlflow.log_artifact(str(artifact))

        return run.info.run_id

# --------------------------------------------------------------------------------------- #
# Registro da Logistic Regression (modelo de recompensa)
# --------------------------------------------------------------------------------------- #

def registrar_reward_model(arm, params, metrics, artifacts=None):

    configurar_mlflow()

    mlflow.set_experiment(
        "reward_models_logistic_regression"
    )

    with mlflow.start_run(
        run_name=f"LogisticRegression_{arm}"
    ) as run:

        mlflow.log_param("arm", arm)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if artifacts:

            for artifact in artifacts:

                artifact = Path(artifact)

                if artifact.exists():
                    mlflow.log_artifact(str(artifact))

        return run.info.run_id


# --------------------------------------------------------------------------------------- #
# Modelo conceitual MLFlow
# --------------------------------------------------------------------------------------- #

# Experiments

# contextual_bandit_lints
# └── LinTS
#     ├── Parameters
#     │   ├── alpha
#     │   ├── l2_lambda
#     │   ├── seed
#     │   └── ...
#     │
#     ├── Metrics
#     │   ├── reward_estimado
#     │   ├── lift_relativo
#     │   ├── match_rate
#     │   └── ...
#     │
#     └── Artifacts
#         ├── lints_bundle.joblib
#         ├── metricas_lints.csv
#         └── ...

# reward_models_logistic_regression
# ├── LogisticRegression_cellular
# │   ├── Parameters
# │   └── Metrics
# │
# └── LogisticRegression_telephone
#     ├── Parameters
#     └── Metrics

# Notebook
#    │
#    │ HTTP
#    ▼
# MLflow Server :5000
#    │
#    ├── métricas/parâmetros
#    │        ↓
#    │   mlops/mlflow.db
#    │
#    └── artefatos
#             ↓
#        Neon Object Storage
#        model-artifacts/mlflow/



            #         ┌─────────────────────────┐
            #         │       Notebook          │
            #         │                         │
            #         │ mlflow.log_param()      │
            #         │ mlflow.log_metric()     │
            #         │ mlflow.log_artifact()   │
            #         └───────────┬─────────────┘
            #                     │
            #                     ▼
            #         ┌─────────────────────────┐
            #         │     MLflow Server       │
            #         │    localhost:5000       │
            #         └───────┬─────────┬───────┘
            #                 │         │
            #   parâmetros    │         │ arquivos
            #   métricas      │         │ .joblib/.csv/.json
            #   runs          │         │
            #                 ▼         ▼
            #         ┌───────────┐   ┌─────────────────────┐
            #         │ SQLite    │   │ Neon Object Storage │
            #         │           │   │                     │
            #         │mlflow.db  │   │ model-artifacts/    │
            #         │           │   │ └── mlflow/         │
            #         └───────────┘   └─────────────────────┘