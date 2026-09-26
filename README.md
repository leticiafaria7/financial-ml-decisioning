# Plataforma de recomendação adaptativa em canais digitais com abordagem Multi-Armed Bandit

*Tech Challenge da Fase 5 do curso de [pós-graduação em Engenharia de Machine Learning FIAP](https://postech.fiap.com.br/curso/machine-learning-engineering/)*

**Link para a aplicação / API:** https://financial-ml-decisioning.onrender.com/
> *📽️ Vídeo com demonstração técnica do projeto (em breve)*

## 🎯 1. Sobre o projeto
O projeto tem como objetivo construir uma plataforma de experimentação adaptativa para ofertas, mensagens ou próximos passos em canais digitais para uma empresa de segmento financeiro usando Multi-Armed Bandit.

A ideia é decidir, em diferentes canais, qual oferta, mensagem ou próximo passo apresentar para cada cliente elegível.

> ### 🎰 Multi-Armed Bandit

É uma abordagem adaptativa que visa identificar comportamentos distintos, equilibrar exploração e explotação e aprender com respostas observadas, sem congelar a decisão em regras estáticas.

O nome desta abordagem vem da ideia de um jogador diante de várias máquinas caça-níqueis, em que cada máquina tem uma alavanca ("arm"); bandit vem da expressão "one-armed bandit", apelido das máquinas caça-níqueis.

Generalizando, multi-armed bandit é um problema em que um tomador de decisões iterativamente seleciona uma de múltiplas opções fixas quando as propriedades de cada opção são apenas parcialmente conhecidas no momento da escolha, e se tornam mais compreendidas à medida em que o tempo passa. Um aspecto fundamental de problemas bandit é que o fato de escolher um "arm" não afeta as propriedades deste ou de outros "arms".

É um problema clássico de reinforcement learning (aprendizado por reforço) que exemplifica o dilema do tradeoff entre exploração e explotação; em contraste ao reinforcement learning geral, as ações selecionadas nos problemas bandit não afetam a distribuição de recompensas nos arms.

No contexto de soluções de marketing, pode ser empregado em alternativa ao uso de regras fixas e testes A/B longos que desperdiçam tráfego e que demoram para reagir a mudanças de contexto, permitindo a personalização responsável de ofertas de acordo com o perfil de comportamento de cada usuário.

Os principais algoritmos usados em soluções multi-armed bandit são:

| Técnica | Descrição | Como pode ser aplicado no contexto de marketing |
| --- | --- | --- |
| **Thompson Sampling** | Baseia-se na atualização de distribuições de probabilidade sobre a performance de cada alternativa, equilibrando exploração e explotação de acordo com a incerteza sobre seus resultados | Selecionar dinamicamente a oferta, mensagem ou próximo passo mais promissor para cada cliente, aprendendo continuamente a partir das respostas observadas |
| **Epsilon-Greedy** | Baseia-se na escolha da alternativa com melhor desempenho conhecido na maior parte das vezes, reservando uma probabilidade ε para explorar outras alternativas | Priorizar a oferta ou mensagem com maior desempenho observado, enquanto uma parcela dos clientes recebe alternativas menos exploradas para descobrir novas oportunidades |
| **UCB (Upper Confidence Bound)** | Baseia-se na combinação do desempenho observado de cada alternativa com um intervalo de confiança que favorece opções com maior incerteza ou menor número de observações | Selecionar ofertas ou mensagens considerando tanto seu desempenho histórico quanto o potencial de alternativas ainda pouco exploradas |


> ### 🎲 Fontes de dados

Os dados utilizados neste projeto foram dispolibilizados pelo perfil do usuário [tunguz](https://www.kaggle.com/tunguz) no site do Kaggle: [Bank Marketing Data Set](https://www.kaggle.com/datasets/tunguz/bank-marketing-data-set)

É uma tabela derivada do estudo publicado no artigo [A data-driven approach to predict the success of bank telemarketing](https://sci-hub.box/10.1016/j.dss.2014.03.001), de Moro at al., 2014.

- São dados de campanhas diretas de marketing (por telefone) de uma instituição bancária portuguesa.
- A tabela tem 41.188 linhas e 21 colunas, sendo que a última informa se o cliente contratou o produto (term depoist), pode ser yes ou no.
- Não há identificação dos clientes, apenas dados de perfil, de data, dados da campanha e dados econômicos do momento da ligação.
- Os dados variam entre maio/2008 e novembro/2010 e estão ordenados por data (apenas mês e dia da semana)

A análise exploratória está disponível no notebook [1_eda.ipynb](notebooks/eda.ipynb)

**Observações: Como a base utilizada não registra qual oferta foi apresentada, o reward usado no protótipo será a conversão do term deposit.**

> **CENÁRIO IDEAL**

Em um cenário ideal para a implementação e avaliação de uma política de Multi-Armed Bandit, a base de dados registraria, para cada decisão:
- o contexto do cliente
- o conjunto de ofertas disponíveis
- a oferta efetivamente selecionada como braço
- a probabilidade de seleção dessa oferta pela política vigente (selection probability/propensity)
- o reward observado após a interação, como a conversão

Essa estrutura permitiria avaliar de forma mais rigorosa diferentes políticas adaptativas e realizar técnicas de avaliação off-policy.

Além disso, seria interessante que a base de dados utilizada fosse derivada de um experimento randomizado, e não de eventos observacionais.

> **CENÁRIO REAL**

Para o escopo simplificado deste projeto, será utilizada a base supracitada (conforme recomendado na descrição do tech challenge) que contém:
- características dos clientes
- informações das campanhas de marketing
- forma de contato (celular ou telefone)
- o resultado da contratação de um depósito a prazo

O principal ponto de limitação para a abordagem multi-armed bandit é que a tabela NÃO registra diferentes ofertas nem as probabilidades históricas de seleção. 

Dessa forma, o projeto deve ser entendido como uma demonstração controlada dos conceitos de Multi-Armed Bandit, na qual os braços e o processo de decisão são definidos ou simulados a partir das informações disponíveis, permitindo comparar estratégias como Thompson Sampling, Epsilon-Greedy e UCB com um baseline determinístico.

Os resultados obtidos, portanto, demonstram o comportamento e o potencial da abordagem adaptativa dentro das premissas do experimento, e não devem ser interpretados como uma estimativa causal do ganho que seria obtido por uma política de ofertas implantada em produção. 

> ### 🦾 Definições do bandit

**Features utilizadas**

| Tipo                   | Feature                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| Perfil                 | Idade                                                            |
| Perfil                 | Faixa etária                                                     |
| Perfil                 | Profissão                                                        |
| Perfil                 | Estado civil                                                     |
| Perfil                 | Escolaridade                                                     |
| Crédito                | Se tem financiamento imobiliário                                 |
| Crédito                | Se tem empréstimo pessoal                                        |
| Data                   | Mês                                                              |
| Data                   | Dia da semana                                                    |
| Campanhas anteriores   | Quantos contatos teve em campanhas anteriores                    |
| Campanhas anteriores   | Resultado da campanha anterior (sucesso, falha ou não existente) |
| Indicadores econômicos | Consumer confidence index — indicador mensal                     |
| Indicadores econômicos | Consumer price index — indicador mensal                          |
| Indicadores econômicos | Employment variation rate — indicador trimestral                 |
| Indicadores econômicos | Euribor 3 month rate — indicador diário                          |
| Indicadores econômicos | Number of employees — indicador trimestral                       |


**Variáveis removidas da tabela original**
- campaign: não faz sentido usar a quantidade de contatos durante a campanha porque isso não necessariamente será usado ao fazer o predict
- duration: não tem como saber qual será a duração da ligação ao fazer o contato
- pdays: a maioria é 999 (equivalente a null)
- default (se tem crédito por padrão): apenas 3 estão como yes, o resto é no ou unknown
- year: se fazemos split temporal, não faz sentido usar o ano (os anos da fração de teste não estarão na fração de treino)

**Braços**
1. Braço 1: oferecer term deposit via celular
2. Braço 2: oferecer sterm deposit via telefone

**Reward** (coluna 'y')
- yes: conversão
- no: não conversão

> ### 🧠 Escolha do algoritmo

O estudo descrito no artigo [Improving Online Marketing Experiments with Drifting Multi-armed Bandits](https://www.scitepress.org/papers/2015/54587/54587.pdf) (Burtini et al. 2015) cita os 3 algoritmos e alguns outros para serem aplicados em um experimento de marketing, mas escolhe testar o Thompson Sampling por ser o estado da arte no tratamento de regression bandits não estacionários (em constante mudança), e porque os resultados dos demais algoritmos (ε-greedy e UCB) tiveram performance muito inferior.

Considerando a base de dados disponível e as aplicações possíveis de cada algoritmo, a abordagem escolhida será o **Thompson Sampling** em sua versão contextual, preferencialmente o Linear Thompson Sampling (LinTS), por combinar uma estratégia de **exploração baseada na incerteza** com a capacidade de **considerar características do cliente** na escolha da ação. 

Embora Epsilon-Greedy seja mais simples de implementar, sua exploração é essencialmente aleatória e depende da definição do parâmetro epsilon, enquanto o UCB também apresenta uma estratégia robusta baseada em incerteza e constitui uma alternativa válida.

O Thompson Sampling, porém, oferece uma combinação adequada de fundamentação teórica, bom desempenho empírico e interpretabilidade para o contexto deste projeto. A utilização de uma versão contextual é particularmente importante, pois permite que variáveis como idade, profissão, histórico de campanhas e contexto econômico influenciem a recomendação, em vez de aprender apenas qual braço apresenta maior conversão média para toda a população. 

Dessa forma, o LinTS será utilizado como política adaptativa principal e comparado a um baseline determinístico, mantendo o escopo do projeto mais simples sem perder a demonstração dos principais conceitos de exploração, explotação e personalização.

> ### 📈 Resultados do modelo

> Em breve

## ⚙️ 2. Etapas do projeto

- Download, leitura e pré-processamento dos dados
- Treino do modelo
    - Definição do baseline
    - Definição de hiperparâmetros do modelo LinTS
    - Treino do modelo LinTS
    - Treino da LogisticRegression para calcular a probabilidade de conversão
    - Criação do Golden Set (exemplos diversos para demonstração da recomendação)
- Uso da ferramenta de versionamento MLFlow localmente
    - Instalação e configuração do MLFlow
    - Registro das métricas 
    - Registro a versão do modelo/política utilizada
- Uso do banco de dados Neon Database
    - Criação de uma conta e de um projeto com object storage
    - Persistência dos artefatos do modelo e tabelas auxiliares de valores de indicadores econômicos para servir os dados através da API
    - Registro dos predicts feitos através de requisições da API
- Desenvolvimento da API
    - Receber dados de um cliente/contexto
    - Gerar a recomendação de braço/oferta
    - Retornar a oferta recomendada
    - Criação de interface para receber inputs do usuário e retornar outputs de recomendação
    - Deploy no render e monitoramento do health com UptimeRobot

## 📐 3. Arquitetura

A arquitetura do projeto como foi construído (on-premise) pode ser observada no diagrama abaixo

![Arquitetura](diagrams/arquitetura.png)

Se o projeto fosse colocado no ar usando os serviços da AWS, seriam utilizados os seguintes recursos:

- Na etapa de fontes de dados e feature engineering, o dataset baixado do Kaggle poderia ser armazenado em um bucket do **Amazon S3,** que funcionaria como data lake bruto (camada raw) e, após o processamento, também guardaria as tabelas mensais de indicadores econômicos e a base já tratada (camada trusted). 

- O processamento de criação da coluna de ano e da divisão temporal do train_test_split poderia ser feito em notebooks ou jobs do **Amazon SageMaker Processing** (ou, para um pipeline mais leve, em uma função **AWS Lambda** ou em um job do **AWS Glue**), lendo os dados diretamente do **S3** e regravando as tabelas processadas também no **S3**. 

- Já a etapa de treino do modelo — definição de baseline, hiperparâmetros, treino do LinTS e da LogisticRegression — seria natural no **Amazon SageMaker Training**, que permite rodar os treinos em instâncias gerenciadas, versionar os experimentos e, combinado ao **SageMaker Model Registry**, substituir ou complementar o papel do MLflow no versionamento dos artefatos do modelo.

- Para a persistência dos dados, o Neon Database poderia ser substituído por um banco gerenciado na AWS, como o **Amazon RDS for PostgreSQL** (equivalente relacional direto) ou o **Amazon DynamoDB**, caso o padrão de acesso aos artefatos e predições fosse mais simples e orientado a chave-valor.

- Os artefatos do modelo (pesos, preprocessador, bandit) continuariam também versionados no **S3**, com o banco guardando metadados e resultados das predições. 

- Na etapa de desenvolvimento e deploy da API, a aplicação Flask poderia ser empacotada em contêiner e implantada no **Amazon ECS** (Fargate) ou no **AWS App Runner**, com o tráfego exposto por um **Application Load Balancer** e, opcionalmente, o **Amazon API Gateway** na frente para gerenciar os endpoints /api/predict e /api/health. 

- O monitoramento, hoje feito pelo UptimeRobot, seria coberto pelo **Amazon CloudWatch** (métricas, logs e alarmes de saúde do serviço) combinado a checagens periódicas via **CloudWatch Synthetics** ou **Route 53 Health Checks**, mantendo a mesma função de verificar continuamente a disponibilidade do endpoint de saúde da API.


## 📁 4. Estrutura do projeto

```
financial-ml-decisioning
├── data/                                           # dados brutos e tratados (ignorado)
│   ├── raw/
│   │   └── bank-marketing-data-set
│   ├── trusted/
│   │   ├── cons_conf_idx_mensal.parquet
│   │   ├── cons_price_idx_mensal.parquet
│   │   ├── emp_var_rate_mensal.parquet
│   │   ├── euribor3m_mensal.parquet
│   │   ├── nr_employed_mensal.parquet
│   │   └── tabela_analitica.parquet
│   └── refined/
├── diagrams/                                       # diagramas arquiteturais
│   └── arquitetura.png
├── mlops/                                          # arquivos de configuração do MLFlow
│   ├── artifacts/
│   ├── __init__.py
│   ├── mlflow.db
│   ├── start_mlflow.sh
│   └── tracking.py
├── models/                                         # artefatos dos modelos treinados (arquivos para inferência, metadados e resultados)
│   ├── results/
│   │   ├── avaliacao_lints_test.csv
│   │   ├── baseline3_classificacao.csv
│   │   ├── comparacao_politicas.csv
│   │   ├── comparacao_por_braco.csv
│   │   ├── comparacao_por_segmento.csv
│   │   ├── metricas_lints.csv
│   │   ├── metricas_reward_models.csv
│   │   ├── tuning_runs_lints.csv
│   │   └── tuning_summary_lints.csv
│   ├── lints_bundle.joblib
│   ├── lints_features.csv
│   ├── lints_metadata.json
│   ├── lints_model.joblib
│   ├── preprocessor.joblib
│   └── reward_models.joblib
├── notebooks/                                      # EDA, treino do modelo e golden set
│   ├── 1_eda.ipynb
│   ├── 2_feature_eng.ipynb
│   ├── 3_model_train.ipynb
│   ├── 4_golden_set.ipynb
│   ├── 5_upload_indicators_neon.ipynb
│   └── 6_teste_api_prod.ipynb
├── src/                                            # criação da API, interface da aplicação e configuração do neon database
│   ├── api/
│   │   ├── api_endpoints.py
│   │   ├── home_backend.py
│   │   └── predict_log.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── neon.py
│   ├── static/
│   │   ├── favicon.svg
│   │   ├── github.svg
│   │   └── swagger.svg
│   ├── templates/
│   │   └── home.html
│   ├── __init__.py
│   └── utils_eda.py
├── tests/                                          # notebooks de teste (não vai para repositório remoto)
│   ├── data_selection.ipynb
│   ├── get_folder_tree.ipynb
│   └── teste_neon.ipynb
├── .env                                            # arquivo com as keys de conexão com o banco (ignorado)
├── .gitignore
├── .neon                                           # credenciais do projeto (ignorado)
├── .python-version                                 # versão do python para render
├── create_table_requests.py                        # criar a tabela no neon para registrar uso do endpoint /predict
├── main.py                                         # arquivo principal da aplicação
├── neon.ts                                         # configuração do neon
├── package-lock.json                               # configuração do neon (ignorado)
├── package.json                                    # configuração do neon
├── README.md
└── requirements.txt                                # libs necessárias para rodar o projeto
```

## 🛠️ 5. Instruções de execução

> *Em construção*

### Requisitos:
- Python 3.11 instalado

### 5.1 Configurar ambiente virtual

- Criar ambiente virtual

```
# se windows:
python -m venv .venv

# se mac ou linux:
python3.11 -m venv .venv
```

- Ativar ambiente virtual

```
# windows
.venv\Scripts\Activate.ps1

# mac ou linux
source .venv/bin/activate
```

- Instalar dependências

```
# windows
pip install -r requirements.txt

# mac ou linux
pip3 install -r requirements.txt
```

- Criar ipykernel para usar o .venv criado nos notebooks

```
# windows
python -m ipykernel install --user --name=venv-decisioning --display-name="Python (venv-decisioning)"

# mac ou linux
python3 -m ipykernel install --user --name=venv-decisioning --display-name="Python (venv-decisioning)"
```

### 5.2 Reproduzir feature engineering nos dados do Kaggle

Passos disponíveis no notebook [2_feature_eng.ipynb](notebooks/6_teste_api_prod.ipynb)

### 5.3 Usar API localmente

Passos disponíveis no notebook [6_teste_api_prod.ipynb](notebooks/6_teste_api_prod.ipynb)

### 5.4 Acessar Swagger

A documentação dos endpoints da API está disponível em https://financial-ml-decisioning.onrender.com/docs/

### 5.5 Configurar Neon Database

1. Acessar o site https://neon.com/
2. Criar uma conta
3. Acessar `+ New project` (para este projeto foi usado o nome `financial-ml-decisioning`)
4. Criar um object storage (para este projeto foi usado o nome `model-artifacts`)
5. No repositório local, criar um .env (colocar no `.gitignore` para não subir para o repositório remoto)
6. No terminal, executar a sequência de comandos (compatíveis com MacOS):

    ``` bash
    # instalar a CLI do Neon (macOS)
    brew install neonctl

    # verificar a versão instalada do Neon CLI
    neonctl --version

    # fazer autenticação da CLI na conta do Neon
    neonctl auth

    # vincular o diretório/repositório local a um projeto existente no Neon
    neonctl link

    # inicializar a configuração do Neon no projeto e criar os arquivos de configuração necessários
    neon config init

    # instalar o pacote de configuração do Neon utilizado pelo arquivo neon.ts
    npm install @neon/config

    # verificar o plano de configuração antes de aplicar alterações no Neon
    neonctl config plan

    # aplicar no projeto Neon as alterações definidas na configuração local
    neonctl config deploy

    # obter as variáveis de ambiente necessárias para utilizar o Object Storage
    # e adicioná-las ao arquivo .env local
    neonctl env pull --service object-storage

    # listar somente os nomes das variáveis existentes no .env, sem exibir seus valores ou credenciais
    grep -o '^[A-Za-z_][A-Za-z0-9_]*=' .env

    # verificar se as variáveis do Object Storage foram carregadas no ambiente local
    # sem imprimir os valores secretos
    grep -E '^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_ENDPOINT_URL_S3|AWS_REGION|NEON_S3_BUCKET)=' .env | cut -d= -f1

    # nunca versionar o arquivo .env, pois ele contém credenciais e outros valores sensíveis
    echo ".env" >> .gitignore
    ```

7. Para fazer upload de tabelas: exemplo no notebook [5_upload_indicators_neon.ipynb](notebooks/5_upload_indicators_neon.ipynb)

### 5.6 Configurar MLFlow

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

## 🚀 6. Evolução do projeto

Limitações do projeto:
- A base utilizada não registra qual oferta foi apresentada, então a avaliação de um bandit precisa explicitar como os "braços" serão definidos e como o reward será observado ou simulado
- A base tem dados muito antigos (2008 a 2010), e os indicadores econômicos mudaram muito desde então. Seria necessário usar uma tabela mais atual para previsões mais precisas para os dias de hoje
- A base não tem todos os meses de todos os anos e o volume de observações varia ao longo dos meses

Ideias para **refinamento do modelo**:
- Usar uma base de dados mais robusta (como explicado na parte de "Observações" da seção Fontes de dados)
- Implementar logging e monitoramento de decisões
- Criar experimento A/B ou contextual bandit real para coletar dados de novas ofertas
- Usar braços de comunicação e oferta, como: abordagem padrão, abordagem personalizada por perfil, abordagem de follow-up, variações de mensagem/argumentação comercial
- Evoluir de conversão binária para reward econômico, considerando valor da contratação e eventual custo da ação
- Exploração mais exaustiva do conjunto de variáveis utiizadas e hiperparâmetros para melhorar a performance

Ideias para **refinamento de infraestrutura**:
- Implementar o projeto em uma cloud (AWS por exemplo)
- Criar processo de retreinamento/atualização automática
- Monitorar drift dos dados e alteração do comportamento por período
- Criar uma tabela para registrar todas as requisições de todos os endpoints
- Explorar mais a ferramenta MLFlow para versionamento