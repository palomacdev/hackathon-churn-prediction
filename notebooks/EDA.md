# 📊 Documentação da Análise Exploratória de Dados
## Projeto: Sistema Inteligente de Análise Financeira — Churn Prediction
**Hackathon de Dados | Tema 4 — Decision Intelligence**
Dataset: Bank Customer Churn Prediction (Kaggle — gauravtopre)

---

## 1. Introdução e Contexto

O objetivo deste projeto é prever o **churn de clientes bancários** — ou seja, identificar com antecedência quais clientes têm maior probabilidade de encerrar seu relacionamento com o banco. Essa predição permite que a instituição tome ações proativas de retenção antes que o cancelamento aconteça, reduzindo perda de receita.

A Análise Exploratória de Dados (EDA) é a **primeira e mais importante etapa** do pipeline de dados. Antes de treinar qualquer modelo, precisamos:

- Entender a estrutura e qualidade dos dados disponíveis
- Identificar padrões e relações entre as variáveis e o churn
- Detectar problemas como nulos, outliers e desbalanceamento de classes
- Gerar hipóteses que guiarão as decisões de feature engineering e modelagem
- Produzir evidências estatísticas que justifiquem nossas escolhas técnicas

Uma EDA bem feita é o que separa um modelo que "funciona" de um modelo que **resolve o problema de negócio**.

---

## 2. Sobre o Dataset

| Atributo | Valor |
|---|---|
| **Fonte** | Kaggle — gauravtopre/bank-customer-churn-dataset |
| **Registros** | 10.000 clientes |
| **Variáveis** | 12 colunas |
| **Valores nulos** | 0 (dataset completo) |
| **Duplicatas** | 0 |
| **Variável alvo** | `churn` (0 = ativo, 1 = cancelou) |
| **Taxa de churn** | ~20,4% |

### Dicionário de Variáveis

| Coluna | Tipo | Descrição |
|---|---|---|
| `customer_id` | int | Identificador único do cliente |
| `credit_score` | int | Score de crédito (300–850) |
| `country` | string | País do cliente (France, Germany, Spain) |
| `gender` | string | Gênero (Male, Female) |
| `age` | int | Idade do cliente |
| `tenure` | int | Anos de relacionamento com o banco (0–10) |
| `balance` | float | Saldo em conta |
| `products_number` | int | Número de produtos contratados (1–4) |
| `credit_card` | int | Possui cartão de crédito (0/1) |
| `active_member` | int | É membro ativo (0/1) |
| `estimated_salary` | float | Salário estimado |
| `churn` | int | **Variável alvo**: cancelou (1) ou não (0) |

---

## 3. Estrutura da Análise

A EDA foi estruturada em **11 seções progressivas**, seguindo a lógica de complexidade crescente: primeiro entendemos cada variável isoladamente, depois suas relações entre si e com o churn, e por fim sintetizamos os achados em insights acionáveis.

```
Dataset bruto
    │
    ├── Inspeção inicial (qualidade, estrutura)
    ├── Estatística descritiva (tendência, dispersão)
    ├── Variável alvo (distribuição, desbalanceamento)
    │
    ├── Análise Univariada
    │       ├── Numéricas (distribuição, skewness)
    │       └── Categóricas (contagem, proporção)
    │
    ├── Análise Bivariada
    │       ├── Feature × Churn (diferenças de distribuição)
    │       ├── Teste estatístico Mann-Whitney U
    │       └── Taxa de churn por categoria
    │
    ├── Correlações (Pearson, heatmap)
    ├── Análise Multivariada (cruzamentos, heatmaps)
    ├── Detecção de Outliers (IQR)
    │
    └── Decisões de Modelagem
```

---

## 4. Seção 1 — Setup e Configuração Visual

### O que foi feito
Carregamento das bibliotecas e definição de uma **paleta de cores consistente** para todos os gráficos do notebook.

### Por que fizemos assim
A consistência visual não é apenas estética — ela é funcional. Ao usar sempre a mesma cor para cada conceito (verde = ativo/positivo, coral = churn/alerta), criamos uma linguagem visual que torna os gráficos imediatamente interpretáveis sem precisar ler a legenda toda vez.

| Cor | Hex | Uso |
|---|---|---|
| Verde | `#1D9E75` | Clientes ativos, valores positivos |
| Coral | `#E8593C` | Clientes em churn, alertas |
| Azul | `#3B8BD4` | Destaques neutros, distribuições |
| Roxo | `#7F77DD` | Terceiro grupo ou comparativo |
| Cinza | `#B4B2A9` | Elementos secundários |

Também configuramos `rcParams` globais do matplotlib para remover as bordas superior e direita dos gráficos (estilo mais limpo), adicionar grade suave e padronizar tamanhos de fonte.

---

## 5. Seção 2 — Inspeção Inicial

### O que foi feito
- Visualização das primeiras linhas com `df.head(10)`
- Informações técnicas com `df.info()`
- Contagem de valores nulos e duplicatas
- Contagem de valores únicos por coluna
- **Gráfico de completude**: barras horizontais mostrando o percentual de preenchimento de cada coluna

### Por que fizemos assim
A inspeção inicial é obrigatória antes de qualquer análise. Sem ela, corremos o risco de tirar conclusões incorretas de dados com problemas ocultos.

**Por que o gráfico de completude?** Em vez de apenas imprimir um número, o gráfico de barras horizontais permite identificar visualmente colunas problemáticas num relance. Usamos verde para 100% de completude e coral para colunas com nulos — neste dataset, todas as barras saíram verdes, confirmando a alta qualidade dos dados.

**Achado importante:** O dataset não possui nenhum valor nulo nem duplicata, o que elimina a necessidade de etapas de imputação ou deduplicação. Isso foi documentado explicitamente para transparência com a banca avaliadora.

---

## 6. Seção 3 — Estatística Descritiva

### O que foi feito
- Tabela de estatísticas descritivas estendida com **coeficiente de variação (CV)** e amplitude
- Análise de **assimetria (skewness)** e **curtose (kurtosis)** para todas as features numéricas

### Por que fizemos assim

O `df.describe()` padrão mostra média, desvio padrão, mín/máx e quartis — mas isso não é suficiente para uma EDA profissional. Adicionamos:

**Coeficiente de Variação (CV = desvio/média × 100):** Mede a dispersão relativa, permitindo comparar a variabilidade de features com escalas completamente diferentes. Um CV de 80% no `balance` vs 30% no `age` indica que o saldo varia muito mais entre clientes do que a idade.

**Assimetria (Skewness):** Indica se a distribuição é simétrica ou tem uma "cauda" longa para um dos lados.
- `|skew| < 0.5`: distribuição simétrica ✅
- `0.5 < |skew| < 1.0`: leve assimetria ⚠️
- `|skew| > 1.0`: alta assimetria ❌ — pode indicar outliers ou necessidade de transformação

**Curtose (Kurtosis):** Indica se a distribuição tem caudas pesadas em comparação com a normal. Alta curtose pode indicar outliers concentrados nos extremos.

Essa análise guia decisões posteriores: features com alta assimetria podem se beneficiar de transformações logarítmicas antes do treino.

---

## 7. Seção 4 — Análise da Variável Alvo

### O que foi feito
Três visualizações complementares da distribuição do churn:
1. **Gráfico de barras** com contagem absoluta e percentual
2. **Gráfico de pizza** com proporção das classes
3. **Waffle chart** — cada bloco representa 1% da base

### Por que escolhemos esses três gráficos

O gráfico de barras é o mais preciso para leitura de valores absolutos. A pizza é intuitiva para comunicar proporções rapidamente. O waffle chart foi escolhido como diferencial visual: ele torna o desbalanceamento tangível e memorável — ao ver 20 blocos vermelhos contra 80 verdes, qualquer pessoa entende imediatamente que apenas 1 em cada 5 clientes cancela.

**Por que isso importa para a modelagem:**

A taxa de churn de ~20,4% indica um dataset **desbalanceado** (razão 4:1). Isso tem consequências diretas:

- Um modelo que classifica **tudo como "ativo"** atingiria 80% de acurácia sem aprender nada
- Por isso, **acurácia não é a métrica principal** do projeto — usamos **AUC-ROC**
- AUC-ROC mede a capacidade discriminatória do modelo independentemente do threshold e é robusta ao desbalanceamento
- O desbalanceamento também justifica o uso de `stratify=y` no split treino/teste e o ajuste de threshold na modelagem

---

## 8. Seção 5 — Análise Univariada: Numéricas

### O que foi feito
Para cada uma das 6 features numéricas (`credit_score`, `age`, `tenure`, `balance`, `products_number`, `estimated_salary`):
- **Histograma com sobreposição de KDE** (Kernel Density Estimation)
- Linhas verticais marcando **média** (coral) e **mediana** (verde)
- Anotação do valor de **skewness**

Análise adicional e dedicada ao `balance` com dois gráficos: distribuição completa e distribuição apenas dos clientes com saldo positivo.

### Por que histograma + KDE

O histograma puro depende do número de bins escolhido — com poucos bins, perdemos detalhes; com muitos, fica ruidoso. A curva KDE suaviza essa dependência e mostra a forma da distribuição de maneira mais robusta. Usar os dois juntos é a prática padrão em EDA profissional.

**Por que marcar média e mediana separadamente?** A distância entre média e mediana é um indicador de assimetria. Quando diferem muito, há evidência de outliers puxando a média para um dos lados — e a mediana é a medida mais confiável de tendência central nesses casos.

### Por que análise dedicada ao `balance`

O `balance` apresentou uma distribuição **bimodal**: um pico muito alto em zero e uma distribuição aproximadamente normal para valores positivos. Isso é um padrão importante porque os dois grupos podem ter comportamentos de churn completamente diferentes. Confirmamos que **29,6% dos clientes têm saldo zero** — e a análise bivariada posterior mostrou que clientes com saldo positivo têm maior taxa de churn.

---

## 9. Seção 6 — Análise Univariada: Categóricas

### O que foi feito
Gráficos de barras verticais para as 4 variáveis categóricas/binárias:
- `country` (France, Germany, Spain)
- `gender` (Male, Female)
- `credit_card` (0/1)
- `active_member` (0/1)

### Por que barras e não pizza

Para variáveis categóricas com mais de 2 categorias, gráficos de barras são mais precisos para comparação de valores. Pizzas ficam difíceis de ler quando os fatias têm tamanhos similares. As barras permitem adicionar rótulos com valores absolutos e percentuais simultaneamente.

**Achados desta seção:**
- França concentra ~50% dos clientes, seguida de Alemanha (~25%) e Espanha (~25%)
- A base é ligeiramente masculina (~54% homens)
- ~71% dos clientes possuem cartão de crédito
- ~51% são membros ativos — proporção quase equilibrada

---

## 10. Seção 7 — Análise Bivariada

Esta é a seção mais crítica do EDA — onde identificamos quais features têm **poder preditivo real** sobre o churn.

### Gráfico 1: Distribuição de numéricas por status de churn

**O que é:** Para cada feature numérica, dois histogramas sobrepostos (ativo em verde, churned em coral) com curvas KDE, e o resultado de um **teste estatístico Mann-Whitney U**.

**Por que Mann-Whitney U e não t-test?** O teste t pressupõe normalidade das distribuições. Como a análise de skewness revelou que várias features não são normais (especialmente `balance`), optamos pelo Mann-Whitney U, que é **não paramétrico** — não exige nenhuma suposição sobre a forma da distribuição. Ele testa se as duas distribuições são estatisticamente diferentes. Um p-valor < 0,05 indica que a feature discrimina os grupos e é relevante para o modelo.

**Achados:**
- `age` e `active_member` apresentam as diferenças mais visíveis entre ativos e churned
- `estimated_salary` mostrou distribuições quase idênticas entre os grupos — sinal de baixo poder preditivo
- `products_number` revelou um padrão não linear: clientes com 3–4 produtos churnam muito mais

### Gráfico 2: Taxa de churn por variável categórica

**O que é:** Barras horizontais mostrando a taxa de churn (%) para cada categoria, com uma linha tracejada indicando a taxa média geral.

**Por que esse formato?** A taxa relativa é mais informativa do que a contagem absoluta. França tem mais clientes em termos absolutos, mas Alemanha tem a maior **taxa** de churn — e esse é o dado acionável para o banco.

**Achados críticos:**
- Alemanha: ~32% de churn vs média de ~20%
- Mulheres: ~25% vs homens ~16%
- Membros inativos: ~27% vs ativos ~14%

### Gráfico 3: Boxplots

**O que é:** Boxplots lado a lado (ativo vs churned) para cada feature numérica, com anotação da diferença percentual entre as medianas.

**Por que boxplots aqui?** Os histogramas mostram a forma da distribuição; os boxplots mostram a posição central, dispersão e outliers de forma compacta e comparável. A combinação dos dois é mais completa que usar apenas um.

---

## 11. Seção 8 — Correlações

### O que foi feito
1. **Heatmap da matriz de correlação de Pearson** (triângulo inferior, com anotações numéricas)
2. **Ranking de correlação** de cada feature com o churn, em gráfico de barras horizontais

### Por que usamos o triângulo inferior

A matriz de correlação é simétrica — o valor de A×B é igual ao de B×A. Mostrar o triângulo inteiro dobra a informação sem acrescentar nada, além de dificultar a leitura. O triângulo inferior é a convenção acadêmica padrão.

### Por que barras horizontais para correlação com churn

Barras horizontais com divergência em torno do zero facilitam a leitura de correlações positivas e negativas ao mesmo tempo, com a direção sendo visualmente imediata. Usamos coral para correlações positivas (a feature aumenta o churn) e verde para negativas (a feature reduz o churn).

### Interpretação

**Correlação de Pearson mede relações lineares.** Features com correlação alta em valor absoluto são candidatas fortes a features importantes no modelo. No entanto, correlação zero não implica independência — pode haver relações não lineares que só a modelagem captura.

**Achados:**
- `age` e `active_member` têm as maiores correlações absolutas com churn
- `products_number` tem correlação positiva — mas a análise multivariada revelou que a relação é não linear (clientes com 2 produtos têm baixo churn, mas com 3–4 têm alto)
- `estimated_salary` e `tenure` têm correlação próxima de zero, sugerindo baixo poder preditivo linear

**Multicolinearidade:** Nenhum par de features apresentou correlação > 0,9, o que indica ausência de multicolinearidade severa. Isso é positivo — todas as features carregam informação independente.

---

## 12. Seção 9 — Análise Multivariada

### Gráfico 1: Taxa de churn por País × Gênero (barras agrupadas)

**Por que:** Verificar se o efeito do gênero é consistente em todos os países ou se há uma **interação** entre as duas variáveis. Se mulheres alemãs churnam muito mais que mulheres francesas, isso é um insight que análises univariadas não capturam.

**Achado:** O padrão de mulheres churning mais que homens se repete nos três países, mas é mais pronunciado na Alemanha.

### Gráfico 2: Scatter — Idade × Produtos × Churn

**Por que:** Visualizar simultaneamente três variáveis: duas numéricas nos eixos e o status de churn na cor dos pontos. Permite identificar **regiões de risco** no espaço feature.

**Achado:** Clientes mais velhos (> 40) com poucos produtos concentram muitos pontos vermelhos (churned).

### Gráfico 3: Heatmap — Taxa de churn por Faixa Etária × Nº de Produtos

**Por que:** Este é o gráfico multivariado mais informativo do notebook. O heatmap com escala de cor (verde = baixo churn, vermelho = alto churn) permite identificar combinações críticas de duas variáveis ao mesmo tempo.

**Achado crítico:** Clientes com **3 produtos e 41–60 anos** têm taxas de churn superiores a 80% em alguns segmentos. Isso é uma descoberta que nenhuma análise univariada revelaria e tem implicações diretas para estratégias de retenção.

### Gráfico 4: Balance × Churn

**Por que:** Retomar a análise do `balance` no contexto bivariado, confirmando se a bimodalidade tem relação com o churn.

**Achado:** Clientes com saldo positivo churnam ~27%, enquanto clientes com saldo zero churnam ~17%. Contra-intuitivamente, ter saldo no banco não reduz o churn — pode indicar que esses clientes têm maior mobilidade financeira e mais facilidade de trocar de banco.

---

## 13. Seção 10 — Detecção de Outliers

### O que foi feito
- **Tabela IQR**: para cada feature numérica, calculamos Q1, Q3, IQR e os limites de outlier (Q1 − 1.5×IQR e Q3 + 1.5×IQR), com contagem e percentual de registros fora dos limites
- **Boxplots** com os outliers destacados em coral

### Por que o método IQR e não Z-score

O Z-score pressupõe distribuição normal para definir outliers como ±2 ou ±3 desvios padrão. Como várias features do dataset não são normais (alto skewness), o método IQR é mais robusto — ele é baseado em percentis e funciona bem em qualquer distribuição.

**Regra IQR:** Um valor é outlier se for menor que Q1 − 1,5×IQR ou maior que Q3 + 1,5×IQR.

### Decisão sobre tratamento

Para modelos baseados em árvores de decisão (Random Forest, Gradient Boosting, XGBoost), **outliers têm pouco impacto** — as árvores fazem splits baseados em pontos de corte e não são influenciadas pela magnitude absoluta dos valores. Por essa razão, optamos por **não remover os outliers** antes da modelagem, preservando informação real dos clientes.

A única exceção seria para a Regressão Logística, que é sensível à escala — mas ela já recebe dados normalizados via `StandardScaler`, o que atenua o efeito dos outliers.

---

## 14. Seção 11 — Insights e Conclusões

### O que foi feito
Um gráfico de ranking de importância proxy das features.

### Gráfico: Importância Proxy das Features

**O que é:** Um ranking combinando a correlação absoluta com o churn e a diferença normalizada de médias entre ativos e churned, como estimativa de importância **antes** da modelagem.

**Por que é útil:** Permite comparar features de naturezas diferentes (numéricas e binárias) numa escala comum. Também serve como validação: se as features mais importantes no proxy forem diferentes das mais importantes no modelo treinado, isso é um sinal de relações não lineares que merecem investigação.

---

## 15. Decisões Técnicas Justificadas

| Decisão | Alternativa Considerada | Justificativa da Escolha |
|---|---|---|
| Mann-Whitney U para teste de hipótese | t-test de Student | Features não são normais (skewness alto); Mann-Whitney é não paramétrico |
| Correlação de Pearson | Correlação de Spearman | Pearson é padrão para exploração inicial; Spearman seria mais robusto para não-lineares, mas Pearson é mais interpretável |
| Triângulo inferior no heatmap | Matriz completa | Elimina redundância visual e segue convenção acadêmica |
| Waffle chart para variável alvo | Apenas pizza ou barras | Torna o desbalanceamento tangível e mais memorável visualmente |
| Não remover outliers | Remoção por IQR ou clipping | Modelos de árvore são robustos a outliers; remover pode perder informação real |
| KDE sobre histograma | Apenas histograma | KDE remove dependência da escolha de bins e mostra forma real da distribuição |
| Análise dedicada para `balance` | Tratar igual às outras features | Distribuição bimodal detectada exige análise separada dos dois grupos |

---

## 16. Implicações para a Modelagem

Os achados do EDA levam diretamente às seguintes decisões na etapa de modelagem:

**1. Métrica principal: AUC-ROC**
O desbalanceamento de classes (~20% churn) torna a acurácia uma métrica enganosa. AUC-ROC é robusta ao desbalanceamento e mede a capacidade discriminatória real do modelo.

**2. Ajuste de threshold**
O threshold padrão de 0,5 resulta em recall baixo para a classe churn (~49%). Como o custo de falso negativo (cliente que vai churnar e não é identificado) é maior que o de falso positivo (cliente abordado desnecessariamente), ajustamos o threshold para maximizar o F1-score da classe churn.

**3. Feature engineering sugerida**
- Criar flag binária `tem_balance` (0/1) para capturar a bimodalidade do saldo
- Criar `faixa_etaria` como variável categórica (18–30, 31–40, 41–50, 51–60, 61+)
- O produto cruzado `faixa_etaria × products_number` captura o padrão de risco detectado no heatmap multivariado

**4. Features para manter**
Com base na análise bivariada e de correlações, as features mais relevantes são:
`age`, `active_member`, `balance`, `products_number`, `credit_score`, `country`, `gender`

**5. Features de baixo poder preditivo**
`estimated_salary` e `tenure` mostraram distribuições similares entre ativos e churned e correlação próxima de zero. Podem ser mantidas (modelos de árvore selecionam naturalmente) ou removidas para simplificar o modelo.

**6. Normalização apenas para Regressão Logística**
Modelos baseados em árvores não precisam de normalização. O `StandardScaler` é aplicado apenas para a Regressão Logística, evitando transformação desnecessária para os outros modelos.

---
