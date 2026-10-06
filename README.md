# Churn Prediction Intelligence

**Decision Intelligence project for bank customer churn — from raw data to retention-oriented insights.**

Python · SQLite · scikit-learn · XGBoost · Power BI

> Hackathon de Dados · Tema 4 — Decision Intelligence

## 📌 Sobre o Projeto

Este projeto foi desenvolvido para o **Hackathon de Dados**, com foco em **Decision Intelligence (Tema 4)**: previsão de churn de clientes bancários com suporte à tomada de decisão baseada em dados.

A solução cobre o pipeline completo de dados:

```
CSV bruto → ETL → SQLite → EDA → ML → Score → Dashboard → Insights
```

**Dataset:** [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) — Kaggle  
**10.000 clientes · 12 variáveis · ~20% de taxa de churn**

---

## 🎯 Resultados

| Métrica | Valor |
|---|---|
| **AUC-ROC** (Gradient Boosting) | **0.8756** |
| Recall da classe Churn (threshold ajustado) | **~68%** |
| F1-Score Churn | **0.61** |
| Acurácia geral | **87%** |
| Validação cruzada (5-fold) | **0.8625 ± 0.0042** |

---

## 🏗️ Arquitetura da Solução

```
dados/
│   Bank Customer Churn Prediction.csv   ← fonte original (Kaggle)
│
docs/
│   insights.md                         ← insights e ações recomendadas
│
etl/
│   etl_churn.py                         ← pipeline Extract → Transform → Load
│
notebooks/
│   EDA_Churn.ipynb                      ← análise exploratória completa
│
modelo/
│   modelo_churn.ipynb                   ← treino, avaliação e scoring
│   feature_importance.png               ← importância das features
│   metricas_modelo.png                  ← métricas e avaliação
│   clientes_score_powerbi.csv           ← scores exportados para Power BI
│
sql/
│   queries.sql                          ← 25 queries analíticas
│
powerbi/
│   dashboard.pbix                       ← painel 
│
requirements.txt
.gitignore
```

---

## 🗄️ Modelagem Dimensional — Star Schema

O ETL cria um banco SQLite com **Star Schema** completo:

```
        dim_cliente
            │
dim_pais ───┼─── fato_churn ───┬─── dim_produto
            │                  │
        dim_risco ─────────────┘
```

| Tabela | Descrição |
|---|---|
| `fato_churn` | Tabela central com medidas e chaves estrangeiras |
| `dim_cliente` | Atributos demográficos e financeiros do cliente |
| `dim_pais` | País e região geográfica |
| `dim_produto` | Perfil de contratação e flag de risco |
| `dim_risco` | Faixas de risco com score e cor de alerta |

**4 views analíticas prontas:**
- `vw_cliente_completo` — JOIN de todas as tabelas
- `vw_kpi_por_pais` — taxa de churn, score e balance por país
- `vw_kpi_por_faixa_etaria` — KPIs segmentados por idade
- `vw_alto_risco` — clientes com faixa Alto ou Crítico

---

## 🔍 Principais Insights do EDA

| Nível | Insight |
|---|---|
| 🔴 Alto | Alemanha tem **32% de churn** vs média de 20% |
| 🔴 Alto | Membros inativos churnam **2x mais** que ativos |
| 🔴 Alto | Clientes com **3–4 produtos** têm churn elevado (padrão contra-intuitivo) |
| 🟠 Médio | Faixa etária **41–60 anos** concentra maior risco |
| 🟠 Médio | Clientes com **saldo positivo** churnam mais (~27% vs ~17%) |
| 🟡 Baixo | `estimated_salary` e `credit_card` têm baixo poder preditivo |

---

## 🤖 Modelos Avaliados

| Modelo | AUC-ROC |
|---|---|
| Gradient Boosting | **0.8756** ✅ Escolhido |
| XGBoost | 0.8752 |
| Random Forest | 0.8736 |
| Regressão Logística | 0.7817 |

**Por que Gradient Boosting?**  
Melhor AUC-ROC, estável na validação cruzada (σ = 0.0042) e com excelente equilíbrio entre precision e recall após ajuste de threshold. Ensemble sequencial que corrige erros iterativamente — ideal para dados tabulares financeiros.

**Ajuste de threshold:**  
O threshold padrão (0.5) resultava em recall de ~49% para a classe churn. Usando a curva Precision-Recall, encontramos o threshold ótimo (~0.32–0.35) que eleva o recall para **~68%** — detectando mais clientes em risco antes que cancelem.

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/palomacdev/hackathon-churn-prediction.git
cd hackathon-churn-prediction
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Baixe o dataset
Acesse [kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) e salve o CSV em `dados/`.

### 4. Execute o ETL

> **Nota de reprodutibilidade:** o script de ETL atualmente referencia o caminho usado no ambiente original de desenvolvimento (GitHub Codespaces). Antes de executá-lo em outro ambiente, ajuste `CSV_PATH` em `etl/etl_churn.py` para apontar para o CSV baixado.

```bash
cd etl
python etl_churn.py
# Gera: churn_dw.db e etl_churn.log
```

### 5. Rode o notebook de EDA
```bash
cd notebooks
jupyter notebook EDA_Churn.ipynb
```

### 6. Treine e avalie os modelos

> O notebook de modelagem também referencia o caminho original do dataset em Codespaces. Ajuste a chamada `pd.read_csv(...)` para o caminho local antes da execução.

```bash
cd modelo
jupyter notebook modelo_churn.ipynb
```

O notebook compara Regressão Logística, Random Forest, Gradient Boosting e XGBoost, avalia AUC-ROC e Precision-Recall, ajusta o threshold de decisão e exporta os scores usados na camada de BI.

---

## 📦 Dependências

```txt
pandas · numpy · scikit-learn · xgboost · joblib
matplotlib · seaborn · scipy
shap · nbformat · ipykernel
sqlite3 (built-in)
```

---

## 📁 Artefatos de execução

O pipeline produz artefatos de execução. Banco SQLite, logs e modelos serializados são ignorados pelo Git; o CSV de scores usado pelo dashboard está versionado neste repositório como artefato do projeto.

| Arquivo | Gerado por |
|---|---|
| `dados/churn_dw.db` | `etl/etl_churn.py` |
| `etl/etl_churn.log` | `etl/etl_churn.py` |
| `modelo/clientes_score_powerbi.csv` | notebook `modelo/modelo_churn.ipynb` (versionado para consumo pelo Power BI) |

---

<div align="center">

**Hackathon de Dados · Tema 4 — Decision Intelligence**  
*Pipeline: CSV → ETL → EDA → ML → Score → Dashboard → Insights*

</div>
