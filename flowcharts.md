# Flowcharts — Churn Prediction Pipeline
> Diagramas técnicos do projeto. 

---

## 1. EDA — Análise Exploratória de Dados

```mermaid
flowchart TD
    A([🗂️ Bank Customer Churn\nPrediction.csv\n10.000 registros · 12 colunas]) --> B

    subgraph SEC1["🔎 INSPEÇÃO INICIAL"]
        B[Shape · Tipos · Dtypes] --> C{Nulos?}
        C -->|Sim| D[⚠️ Documentar\ne tratar]
        C -->|Não ✅| E{Duplicatas?}
        D --> E
        E -->|Sim| F[Remover duplicatas]
        E -->|Não ✅| G[Validação de domínios\nchurn 0/1 · age 18–100\nproducts 1–4]
    end

    G --> H

    subgraph SEC2["📊 ESTATÍSTICA DESCRITIVA"]
        H[Média · Mediana · Desvio Padrão] --> I[Coeficiente de Variação]
        I --> J[Skewness · Kurtosis\nAnálise de normalidade]
    end

    J --> K

    subgraph SEC3["🎯 VARIÁVEL ALVO"]
        K[Distribuição do Churn\n20,4% vs 79,6%] --> L[Desbalanceamento 4:1\nAUC-ROC como métrica\nestratify no split]
    end

    L --> M & N

    subgraph SEC4["📈 UNIVARIADA"]
        M[Numéricas\nHistograma + KDE\nMédia vs Mediana\nAnálise de balance bimodal]
        N[Categóricas\ncountry · gender\ncredit_card · active_member]
    end

    M --> O
    N --> O

    subgraph SEC5["🔗 BIVARIADA"]
        O[Distribuição por status\nAtivo vs Churned] --> P[Teste Mann-Whitney U\np-valor por feature]
        P --> Q[Taxa de churn\npor categoria]
        Q --> R[Boxplots comparativos\nΔ medianas]
    end

    R --> S

    subgraph SEC6["🌡️ CORRELAÇÕES"]
        S[Matriz de Pearson\nHeatmap triângulo inferior] --> T[Ranking de correlação\ncom churn\nVerificação de\nmulticolinearidade]
    end

    T --> U

    subgraph SEC7["🔀 MULTIVARIADA"]
        U[País × Gênero] --> V[Faixa etária × Produtos\nHeatmap de churn %]
        V --> W[Balance × Churn\nImpacto do saldo zero]
    end

    W --> X

    subgraph SEC8["⚠️ OUTLIERS"]
        X[Método IQR\nQ1 - 1.5×IQR · Q3 + 1.5×IQR] --> Y{Outliers\nsignificativos?}
        Y -->|Sim| Z[Documentar · manter\nÁrvores são robustas]
        Y -->|Não| AA[✅ Dados limpos]
        Z --> AA
    end

    AA --> BB

    subgraph SEC9["💡 INSIGHTS"]
        BB[Alemanha: 32% churn\nvs média 20%] 
        CC[Inativos churnam 2x mais]
        DD[3+ produtos = risco alto\npadrão contra-intuitivo]
        EE[Faixa 41–60 anos crítica]
        FF[Saldo positivo não\ngarante retenção]
    end

    BB & CC & DD & EE & FF --> GG([🚀 Features selecionadas\npara modelagem\nage · active_member · balance\nproducts_number · credit_score])

    style A fill:#1a2d3d,stroke:#1D9E75,color:#fff
    style GG fill:#1a2d3d,stroke:#1D9E75,color:#fff
    style SEC1 fill:#0f1929,stroke:#3B8BD4,color:#9ca3af
    style SEC2 fill:#0f1929,stroke:#7F77DD,color:#9ca3af
    style SEC3 fill:#0f1929,stroke:#E8593C,color:#9ca3af
    style SEC4 fill:#0f1929,stroke:#3B8BD4,color:#9ca3af
    style SEC5 fill:#0f1929,stroke:#1D9E75,color:#9ca3af
    style SEC6 fill:#0f1929,stroke:#7F77DD,color:#9ca3af
    style SEC7 fill:#0f1929,stroke:#F59E0B,color:#9ca3af
    style SEC8 fill:#0f1929,stroke:#E8593C,color:#9ca3af
    style SEC9 fill:#0f1929,stroke:#1D9E75,color:#9ca3af
```

---

## 2. ETL — Pipeline e Consumo do Banco

```mermaid
flowchart TD
    SRC([📄 Bank Customer Churn\nPrediction.csv]) --> E

    subgraph EXTRACT["📥 E — EXTRACT"]
        E[Leitura do CSV\npd.read_csv] --> EV1{Schema\nválido?}
        EV1 -->|Não| ERR1[❌ FileNotFoundError\nValida colunas esperadas]
        EV1 -->|Sim ✅| EV2{Nulos?}
        EV2 -->|Sim| EV3[⚠️ Registra no log]
        EV2 -->|Não ✅| EV4{Duplicatas\ncustomer_id?}
        EV3 --> EV4
        EV4 -->|Sim| EV5[⚠️ Registra no log]
        EV4 -->|Não ✅| EV6[Validação de domínios\nchurn · credit_card\nactive_member · products]
        EV5 --> EV6
    end

    EV6 --> T

    subgraph TRANSFORM["🔄 T — TRANSFORM"]
        T[Limpeza de tipos\nstr.strip · Title Case\nastype int/float] --> T2

        T2[Feature Engineering] --> T2A[faixa_etaria\npd.cut em 5 grupos]
        T2 --> T2B[faixa_credito\nMuito Baixo → Excelente]
        T2 --> T2C[faixa_tenure\nNovo → Fiel]
        T2 --> T2D[tem_balance\nflag binária 0/1]
        T2 --> T2E[score_risco_heuristico\nInsights do EDA · escala 0–100]

        T2A & T2B & T2C & T2D & T2E --> T3[Criação das dimensões]

        T3 --> D1[dim_pais\npais_id · pais_nome · regiao]
        T3 --> D2[dim_produto\nproduto_id · perfil\nflag_alto_risco_churn]
        T3 --> D3[dim_cliente\ncliente_id · atributos\ndemográficos e financeiros]
        T3 --> D4[dim_risco\nrisco_id · faixa · score\nmin/max · cor_alerta]
        D1 & D2 & D3 & D4 --> F1[fato_churn\nfato_id · FKs · medidas\nbalance · score · churn]
    end

    F1 --> L

    subgraph LOAD["💾 L — LOAD"]
        L[Cria churn_dw.db\nSQLite] --> L2[DDL explícito\nCREATE TABLE com\nPRIMARY KEY · FK\nCHECK constraints]
        L2 --> L3[Carga ordenada\nDimensões → Fato\nrespeitando FK constraints]
        L3 --> L4[Criação de índices\nchurn · cliente_id\npais_id · risco_id]
        L4 --> L5[Views analíticas\nvw_cliente_completo\nvw_kpi_por_pais\nvw_kpi_por_faixa_etaria\nvw_alto_risco]
        L5 --> L6[Verificação pós-carga\nCOUNT por tabela\nLog de auditoria]
    end

    L6 --> DB

    subgraph DB_SCHEMA["🗄️ STAR SCHEMA — churn_dw.db"]
        direction LR
        DB[(fato_churn\n━━━━━━━━━\nfato_id PK\ncliente_id FK\npais_id FK\nqtd_produtos FK\nrisco_id FK\nbalance\nscore_risco\nmembro_ativo\ntem_balance\nchurn)]
        
        DP1[(dim_pais\n━━━━━━━━━\npais_id PK\npais_nome\nregiao)]
        DP2[(dim_produto\n━━━━━━━━━\nproduto_id PK\nqtd_produtos\nperfil\nflag_risco)]
        DP3[(dim_cliente\n━━━━━━━━━\ncliente_id PK\ngenero · idade\nfaixa_etaria\nscore_credito\nfaixa_credito)]
        DP4[(dim_risco\n━━━━━━━━━\nrisco_id PK\nfaixa_risco\nscore_min/max\ncor_alerta)]

        DP1 --> DB
        DP2 --> DB
        DP3 --> DB
        DP4 --> DB
    end

    DB --> CONSUMERS

    subgraph CONSUMERS["📡 CONSUMIDORES DO BANCO"]
        SQL[📝 queries.sql\nAnálises SQL\nSegmentação · KPIs\nReceita em risco]
        PBI[📊 Power BI\nConexão ODBC\nou importação .db]
        STREAM[🖥️ dashboard.py\nStreamlit\npd.read_sql]
        ML[🤖 modelo_churn.py\nFeature engineering\nDataset para treino]
    end

    style SRC fill:#1a2d3d,stroke:#1D9E75,color:#fff
    style EXTRACT fill:#0f1929,stroke:#3B8BD4,color:#9ca3af
    style TRANSFORM fill:#0f1929,stroke:#7F77DD,color:#9ca3af
    style LOAD fill:#0f1929,stroke:#1D9E75,color:#9ca3af
    style DB_SCHEMA fill:#0f1929,stroke:#F59E0B,color:#9ca3af
    style CONSUMERS fill:#0f1929,stroke:#E8593C,color:#9ca3af
    style DB fill:#1a2d3d,stroke:#F59E0B,color:#fff
```

---

## 3. Modelo — Pipeline de Machine Learning

```mermaid
flowchart TD
    IN([🗂️ churn_dw.db\nou CSV fallback\n10.000 registros]) --> PP

    subgraph PREPROC["⚙️ PRÉ-PROCESSAMENTO"]
        PP[Drop customer_id\nnão é feature preditiva] --> PP2[Label Encoding\ncountry · gender\nLabelEncoder] 
        PP2 --> PP3[Separação\nX = features\ny = churn target]
        PP3 --> PP4[Train/Test Split\n70% treino · 30% teste\nstratify=y · random_state=42]
        PP4 --> PP5[StandardScaler\nfit no treino\ntransform no teste\nSÓ para Regressão Logística\nevita data leakage]
    end

    PP5 --> TRAIN

    subgraph TRAIN["🏋️ TREINAMENTO — 4 MODELOS"]
        M1[Regressão Logística\nBaseline · interpretável\nmax_iter=1000]
        M2[Random Forest\n200 árvores · max_depth=10\nBagging · n_jobs=-1]
        M3[Gradient Boosting\n200 estimadores · lr=0.05\nBoosting sequencial]
        M4[XGBoost\n200 rounds · subsample=0.8\ncolsample=0.8 · regularização L1/L2]
    end

    M1 & M2 & M3 & M4 --> EVAL

    subgraph EVAL["📊 AVALIAÇÃO"]
        EVAL1[Classification Report\nPrecision · Recall · F1\npor classe] --> EVAL2[AUC-ROC\nMétrica principal\nrobusta ao desbalanceamento]
        EVAL2 --> EVAL3[Ranking comparativo\nGradient Boosting 0.8756\nXGBoost         0.8752\nRandom Forest   0.8736\nReg. Logística  0.7817]
        EVAL3 --> EVAL4[Validação Cruzada\n5-fold KFold\nMédia ± desvio padrão\nestabilidade do modelo]
    end

    EVAL4 --> BEST

    BEST([🥇 Melhor modelo\nGradient Boosting\nAUC = 0.8756]) --> THRESH & FI

    subgraph THRESH["🎯 AJUSTE DE THRESHOLD"]
        T1[Curva Precision-Recall\npara cada threshold possível] --> T2[Calcula F1 da classe Churn\nem cada ponto]
        T2 --> T3[Threshold ótimo\nmaximiza F1 do Churn\n≈ 0.32–0.35]
        T3 --> T4[Recall churn\n0.49 → 0.65–0.70\nmelhor detecção de risco]
        T4 --> T5[Comparação\nMatriz de confusão\n0.5 vs threshold ótimo]
    end

    subgraph FI["🔍 INTERPRETABILIDADE"]
        F1[Feature Importance\nGradient Boosting\nDecreasing by impurity] --> F2[Top features:\nage · active_member\nbalance · products_number\ncredit_score]
    end

    T5 & F2 --> PROD

    subgraph PROD["🚀 PRODUÇÃO"]
        P1[Aplica modelo\nem toda a base\n10.000 clientes] --> P2[Gera colunas:\nprob_churn 0–1\nscore_risco 0–1000\nfaixa_risco Baixo/Médio/Alto/Crítico]
        P2 --> P3[Serialização\njoblib.dump\nmodelo_churn.pkl\nmodelo + scaler + threshold + features]
        P3 --> P4[Export\nclientes_score_powerbi.csv\npronto para Power BI]
        P3 --> P5[Uso em produção\ncarregar pkl\nprever novos clientes\nsem retreinar]
    end

    P4 --> OUT1([📊 Power BI\nDashboard da diretoria\nscore por cliente])
    P5 --> OUT2([🖥️ Streamlit\nDashboard interativo\nalertas em tempo real])

    style IN fill:#1a2d3d,stroke:#1D9E75,color:#fff
    style BEST fill:#1a2d3d,stroke:#F59E0B,color:#fff
    style OUT1 fill:#1a2d3d,stroke:#1D9E75,color:#fff
    style OUT2 fill:#1a2d3d,stroke:#3B8BD4,color:#fff
    style PREPROC fill:#0f1929,stroke:#3B8BD4,color:#9ca3af
    style TRAIN fill:#0f1929,stroke:#7F77DD,color:#9ca3af
    style EVAL fill:#0f1929,stroke:#1D9E75,color:#9ca3af
    style THRESH fill:#0f1929,stroke:#F59E0B,color:#9ca3af
    style FI fill:#0f1929,stroke:#E8593C,color:#9ca3af
    style PROD fill:#0f1929,stroke:#1D9E75,color:#9ca3af
```