# Plataforma de recomendação adaptativa em canais digitais com abordagem Multi-Armed Bandit

*Tech Challenge da Fase 5 do curso de [pós-graduação em Engenharia de Machine Learning FIAP](https://postech.fiap.com.br/curso/machine-learning-engineering/)*
> *📽️ Vídeo com demonstração técnica do projeto (em breve)*

## 🎯 Sobre o projeto
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
- A tabela tem 41.188 linhas e 21 colunas, sendo que a última informa se o cliente contratou o produto (yes ou no).
- Não há identificação dos clientes, apenas dados de perfil, de data, dados da campanha e dados econômicos do momento da ligação.
- Os dados variam entre maio/2008 e novembro/2010 e estão ordenados por data (apenas mês e dia da semana)

A análise exploratória está disponível no notebook [eda.ipynb](eda.ipynb)

> ### Escolha do algoritmo

O estudo descrito no artigo [Improving Online Marketing Experiments with Drifting Multi-armed Bandits](https://www.scitepress.org/papers/2015/54587/54587.pdf) (Burtini et al. 2015) cita os 3 algoritmos e alguns outros para serem aplicados em um experimento de marketing, mas escolhe testar o Thompson Sampling por ser o estado da arte no tratamento de regression bandits não estacionários (em constante mudança), e porque os resultados dos demais algoritmos (ε-greedy e UCB) tiveram performance muito inferior.

Considerando a base de dados disponível e as aplicações possíveis de cada algoritmo, 

## ⚙️ Funcionalidades

> *Em breve*

## 📐 Arquitetura

> *Em breve*

## 📁 Estrutura do projeto

> *Em breve*

## 🛠️ Instruções de execução

> *Em breve*

## 🚀 Evolução do projeto

> *Em breve*
