<div align="center">

# 🏦 Churn Prediction Intelligence
### Sistema Inteligente de Análise Financeira e Geração de Insights

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**Hackathon de Dados · Tema 4 — Decision Intelligence**

*Prevemos quais clientes vão cancelar e recomendamos ações para retê-los.*

---

</div>

## 📌 Sobre o Projeto

Este projeto foi desenvolvido para o **Hackathon de Dados**, com foco em **Decision Intelligence (Tema 4)**: previsão de churn de clientes bancários com suporte à tomada de decisão baseada em dados.

A solução cobre o pipeline completo de dados:

```
CSV bruto → ETL → SQLite (Star Schema) → EDA → ML → Score → Dashboard → Insights
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
etl/
│   etl_churn.py                         ← pipeline Extract → Transform → Load
│
notebooks/
│   EDA_Churn.ipynb                      ← análise exploratória completa
│   EDA_Documentacao.md                  ← documentação técnica da EDA
│
modelo/
│   modelo_churn.py                      ← treino, avaliação e exportação
│   modelo_churn.pkl                     ← modelo serializado (gerado)
│   clientes_score_powerbi.csv           ← scores para Power BI (gerado)
│
sql/
│   queries.sql                          ← 25 queries analíticas
│
powerbi/
│   dashboard.pbix                       ← painel da diretoria
│
dashboard.py                             ← dashboard Streamlit executivo
flowcharts.md                            ← diagramas técnicos (Mermaid)
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

### 6. Treine o modelo
```bash
cd modelo
jupyter notebook  modelo_churn.ipynb
# Gera: modelo_churn.pkl e clientes_score_powerbi.csv
```

### 7. Suba o dashboard
```bash
streamlit run dashboard.py
```

---

## 📊 Dashboard Executivo

O dashboard Streamlit cobre 5 páginas:

| Página | Conteúdo |
|---|---|
| 🏠 Visão Executiva | KPIs principais, churn por país, faixa etária e produto |
| 🔍 EDA & Perfil | Distribuições, correlações, heatmap multivariado |
| 🤖 Modelo Preditivo | Curva ROC, matriz de confusão, feature importance |
| ⚠️ Alertas de Risco | Threshold ajustável, lista de clientes em risco |
| 💡 Recomendações | Insights, impacto financeiro e plano de ação |

---

## 🗂️ Diagramas Técnicos

Os flowcharts completos do pipeline estão em [`flowcharts.md`](./flowcharts.md):

- **Flowchart 1 — EDA:** 9 seções da análise exploratória
- **Flowchart 2 — ETL:** pipeline E→T→L e Star Schema
- **Flowchart 3 — Modelo:** pré-processamento, treino, avaliação e produção

---

## 📦 Dependências

```txt
pandas · numpy · scikit-learn · xgboost · joblib
matplotlib · seaborn · scipy
streamlit · plotly
nbformat · ipykernel
sqlite3 (built-in)
```

---

## 👩‍💻 Time

| Integrante | Responsabilidade |
|---|---|
| **Paloma** | Modelo de ML · Avaliação · Feature Engineering · ETL |
| | EDA · Análise Exploratória |
| | SQL · Queries Analíticas |
| | Power BI · Dashboard da Diretoria |
| | Dashboard Streamlit |

---

## 📁 Arquivos Gerados (não versionados)

Os arquivos abaixo são gerados pela execução dos scripts e estão no `.gitignore`:

| Arquivo | Gerado por |
|---|---|
| `dados/churn_dw.db` | `etl/etl_churn.py` |
| `etl/etl_churn.log` | `etl/etl_churn.py` |
| `modelo/modelo_churn.pkl` | `modelo/modelo_churn.py` |
| `modelo/clientes_score_powerbi.csv` | `modelo/modelo_churn.py` |

---

<div align="center">

**Hackathon de Dados · Tema 4 — Decision Intelligence**  
*Pipeline: CSV → ETL → EDA → ML → Score → Dashboard → Insights*

</div>