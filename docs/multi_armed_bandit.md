# 🎰 Multi-Armed Bandit

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

> ## 📊 Aplicação no projeto

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

> ## 🧠 Escolha do algoritmo

O estudo descrito no artigo [Improving Online Marketing Experiments with Drifting Multi-armed Bandits](https://www.scitepress.org/papers/2015/54587/54587.pdf) (Burtini et al. 2015) cita os 3 algoritmos e alguns outros para serem aplicados em um experimento de marketing, mas escolhe testar o Thompson Sampling por ser o estado da arte no tratamento de regression bandits não estacionários (em constante mudança), e porque os resultados dos demais algoritmos (ε-greedy e UCB) tiveram performance muito inferior.

Considerando a base de dados disponível e as aplicações possíveis de cada algoritmo, a abordagem escolhida será o **Thompson Sampling** em sua versão contextual, preferencialmente o Linear Thompson Sampling (LinTS), por combinar uma estratégia de **exploração baseada na incerteza** com a capacidade de **considerar características do cliente** na escolha da ação. 

Embora Epsilon-Greedy seja mais simples de implementar, sua exploração é essencialmente aleatória e depende da definição do parâmetro epsilon, enquanto o UCB também apresenta uma estratégia robusta baseada em incerteza e constitui uma alternativa válida.

O Thompson Sampling, porém, oferece uma combinação adequada de fundamentação teórica, bom desempenho empírico e interpretabilidade para o contexto deste projeto. A utilização de uma versão contextual é particularmente importante, pois permite que variáveis como idade, profissão, histórico de campanhas e contexto econômico influenciem a recomendação, em vez de aprender apenas qual braço apresenta maior conversão média para toda a população. 

Dessa forma, o LinTS será utilizado como política adaptativa principal e comparado a um baseline determinístico, mantendo o escopo do projeto mais simples sem perder a demonstração dos principais conceitos de exploração, explotação e personalização.


