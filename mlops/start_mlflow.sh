#!/bin/bash

set -a
source .env
set +a

export MLFLOW_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL_S3"
export AWS_DEFAULT_REGION="$AWS_REGION"

mlflow server \
    --backend-store-uri sqlite:///mlops/mlflow.db \
    --artifacts-destination "s3://${NEON_S3_BUCKET}/mlflow" \
    --host 127.0.0.1 \
    --port 5000


# dar permissão (una única vez)
# chmod +x mlops/start_mlflow.sh

# inicializar
# ./mlops/start_mlflow.sh