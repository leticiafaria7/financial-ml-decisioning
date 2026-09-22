# Plataforma de recomendação adaptativa em canais digitais com abordagem Multi-Armed Bandit

*Tech Challenge da Fase 5 do curso de [pós-graduação em Engenharia de Machine Learning FIAP](https://postech.fiap.com.br/curso/machine-learning-engineering/)*
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
1. Braço 1: term deposit via celular
2. Braço 2: term deposit via telefone

**Reward** (coluna 'y')
- yes: conversão
- no: não conversão

> ### 🧠 Escolha do algoritmo

O estudo descrito no artigo [Improving Online Marketing Experiments with Drifting Multi-armed Bandits](https://www.scitepress.org/papers/2015/54587/54587.pdf) (Burtini et al. 2015) cita os 3 algoritmos e alguns outros para serem aplicados em um experimento de marketing, mas escolhe testar o Thompson Sampling por ser o estado da arte no tratamento de regression bandits não estacionários (em constante mudança), e porque os resultados dos demais algoritmos (ε-greedy e UCB) tiveram performance muito inferior.

Considerando a base de dados disponível e as aplicações possíveis de cada algoritmo, a abordagem escolhida será o **Thompson Sampling** em sua versão contextual, preferencialmente o Linear Thompson Sampling (LinTS), por combinar uma estratégia de **exploração baseada na incerteza** com a capacidade de **considerar características do cliente** na escolha da ação. 

Embora Epsilon-Greedy seja mais simples de implementar, sua exploração é essencialmente aleatória e depende da definição do parâmetro epsilon, enquanto o UCB também apresenta uma estratégia robusta baseada em incerteza e constitui uma alternativa válida.

O Thompson Sampling, porém, oferece uma combinação adequada de fundamentação teórica, bom desempenho empírico e interpretabilidade para o contexto deste projeto. A utilização de uma versão contextual é particularmente importante, pois permite que variáveis como idade, profissão, histórico de campanhas e contexto econômico influenciem a recomendação, em vez de aprender apenas qual braço apresenta maior conversão média para toda a população. 

Dessa forma, o LinTS será utilizado como política adaptativa principal e comparado a um baseline determinístico, mantendo o escopo do projeto mais simples sem perder a demonstração dos principais conceitos de exploração, explotação e personalização.

## ⚙️ 2. Funcionalidades da aplicação

- Receber dados de um cliente/contexto
- Gerar a recomendação de braço/oferta
- Retornar a oferta recomendada
- Retornar os scores/expectativas estimadas por braço para demonstração
- Registrar a versão do modelo/política utilizada

## 📐 3. Arquitetura

> *Em breve*

## 📁 4. Estrutura do projeto

Este repositório está dividido em 4 partes:

1. Notebooks - EDA e demonstração (golden set)
2. Implementação do bandit
3. Construção da API
4. Experimentos de MLFlow

> *Tree - em breve*

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


### 5.2 Iniciar API localmente

### 5.3 Acessar Swagger

## 🚀 6. Evolução do projeto

- Implementar o projeto em uma cloud (AWS por exemplo)
- Usar uma base de dados mais robusta (como explicado na parte de "Observações" da seção Fontes de dados)
- Implantar logging e monitoramento de decisões
- Criar experimento A/B ou contextual bandit real para coletar dados de novas ofertas
- Evoluir de conversão binária para reward econômico, considerando valor da contratação e eventual custo da ação
- Criar processo de retreinamento/atualização automática
- Monitorar drift dos dados e alteração do comportamento por período
- Usar braços de comunicação e oferta, como: abordagem padrão, abordagem personalizada por perfil, abordagem de follow-up, variações de mensagem/argumentação comercial
- Criar uma tabela para registrar todas as requisições de todos os endpoints

Limitações do projeto:
- A base utilizada não registra qual oferta foi apresentada, então a avaliação de um bandit precisa explicitar como os "braços" serão definidos e como o reward será observado ou simulado
- A base tem dados muito antigos (2008 a 2010), e os indicadores econômicos mudaram muito desde então. Seria necessário usar uma tabela mais atual para previsões mais precisas para os dias de hoje
- A base não tem todos os meses de todos os anos e o volume de observações varia ao longo dos meses