## Configuração do MLFlow

1. Adicionar no .env a URI utilizada pela aplicação Python para se comunicar com o servidor MLflow local
```
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

2. As credenciais do Neon Object Storage utilizadas pelo MLflow também devem estar disponíveis no .env; elas são obtidas anteriormente por meio do Neon CLI

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL_S3=...
AWS_REGION=...
NEON_S3_BUCKET=model-artifacts
```

3. Criar a pasta utilizada para armazenar localmente o backend de tracking do MLflow (executar no terminal, na pasta raiz)
```
mkdir -p mlops
```

4. Criar o script mlops/start_mlflow.sh responsável por carregar as variáveis de ambiente e inicializar o servidor local do MLflow (arquivo exemplo em [mlops/start_mlflow.sh](mlops/start_mlflow.sh))

5. Conceder permissão de execução ao script; precisa ser executado apenas uma vez (executar no terminal, na pasta raiz)
```
chmod +x mlops/start_mlflow.sh
```

6. Iniciar o servidor local do MLflow; este comando deve permanecer em execução enquanto os experimentos estiverem sendo registrados (executar no terminal, na pasta raiz)
```
./mlops/start_mlflow.sh
```

7. O MLflow passa a utilizar dois tipos diferentes de armazenamento:
    - mlops/mlflow.db (SQLite local): experimentos, runs, parâmetros, métricas e metadados
    - Neon Object Storage: arquivos registrados como artifacts dos experimentos

8. Acessar a interface local do MLflow após iniciar o servidor
    - abrir no navegador: http://127.0.0.1:5000

9. O arquivo mlops/tracking.py configura os notebooks para enviarem os registros ao servidor MLflow
    - configurar_mlflow() utiliza MLFLOW_TRACKING_URI=http://127.0.0.1:5000

10. Ao executar registrar_lints(), o experimento contextual_bandit_lints é criado ou reutilizado e são registrados os parâmetros, métricas e artifacts referentes ao Linear Thompson Sampling

11. Ao executar registrar_reward_model(), o experimento reward_models_logistic_regression é criado ou reutilizado e é criada uma run para cada braço, por exemplo LogisticRegression_cellular e LogisticRegression_telephone

12. Os parâmetros e métricas enviados por mlflow.log_params() e mlflow.log_metrics() ficam associados às respectivas runs no backend SQLite mlops/mlflow.db

13. Os arquivos enviados por mlflow.log_artifact() são encaminhados pelo servidor MLflow para o diretório mlflow dentro do bucket model-artifacts no Neon Object Storage

    - importante: o dashboard local depende do arquivo mlops/mlflow.db para conhecer experimentos, runs, parâmetros e métricas; o Object Storage sozinho não reconstrói todo o histórico de tracking do MLflow

14. Adicionar o banco SQLite local do MLflow ao .gitignore; ele contém o histórico local das execuções e não deve ser tratado como código-fonte
    - se o arquivo já tiver subido para o Git, executar `git rm --cached mlops/mlflow.db`

15. Para utilizar novamente o MLflow em outro momento, basta iniciar o servidor
```
./mlops/start_mlflow.sh
```

16. Em outro terminal, ativar o ambiente virtual e executar normalmente o notebook de treinamento as chamadas registrar_lints() e registrar_reward_model() serão enviadas ao servidor MLflow em execução

17. Ao finalizar o trabalho, interromper o servidor MLflow no terminal com Ctrl+C