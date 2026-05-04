import pandas as pd
import numpy as np

# CONFIGURAÇÕES 
N_CLIENTES  = 2000   # quantidade de registros
SEED        = 42     # semente aleatória 
ARQUIVO     = 'clientes_churn.csv'

np.random.seed(SEED)
N = N_CLIENTES

# --- Features ---
tempo_contrato       = np.random.randint(1, 61, N)
valor_mensalidade    = np.round(np.random.uniform(49.9, 299.9, N), 2)
num_reclamacoes      = np.random.poisson(1.2, N)
atraso_pagamentos    = np.random.poisson(0.8, N)
produtos_contratados = np.random.randint(1, 6, N)
idade                = np.random.randint(18, 70, N)
score_credito        = np.random.randint(300, 1001, N).astype(float)
variacao_gasto_pct   = np.round(np.random.normal(0, 15, N), 1)

regiao = np.random.choice(
    ['Sudeste', 'Sul', 'Nordeste', 'Centro-Oeste', 'Norte'],
    N, p=[0.42, 0.23, 0.20, 0.10, 0.05]
)
canal_aquisicao = np.random.choice(
    ['Digital', 'Loja física', 'Indicação', 'Telemarketing'],
    N, p=[0.45, 0.25, 0.20, 0.10]
)

# --- Lógica de churn (~18% de taxa) ---
logit = (
     0.5
    - 0.04 * tempo_contrato
    + 0.003 * valor_mensalidade
    + 0.55  * num_reclamacoes
    + 0.70  * atraso_pagamentos
    - 0.35  * produtos_contratados
    - 0.003 * score_credito
    + 0.015 * np.abs(variacao_gasto_pct)
)
prob_churn = 1 / (1 + np.exp(-logit))
cancelou = (np.random.random(N) < prob_churn).astype(int)

# --- Mês de referência (jan/2023 a jun/2024) ---
meses = pd.date_range('2023-01-01', periods=18, freq='MS')
mes_referencia = pd.DatetimeIndex(np.random.choice(meses, N)).strftime('%Y-%m')

# --- Monta o DataFrame ---
df = pd.DataFrame({
    'cliente_id':            [f'CLI{str(i).zfill(5)}' for i in range(1, N + 1)],
    'idade':                 idade,
    'regiao':                regiao,
    'canal_aquisicao':       canal_aquisicao,
    'tempo_contrato_meses':  tempo_contrato,
    'valor_mensalidade':     valor_mensalidade,
    'produtos_contratados':  produtos_contratados,
    'num_reclamacoes':       num_reclamacoes,
    'atraso_pagamentos':     atraso_pagamentos,
    'score_credito':         score_credito,
    'variacao_gasto_pct':    variacao_gasto_pct,
    'mes_referencia':        mes_referencia,
    'cancelou':              cancelou,
})

# --- Introduz nulos realistas (~4%) para exercício de limpeza ---
for col in ['score_credito', 'variacao_gasto_pct', 'canal_aquisicao']:
    idx = np.random.choice(df.index, size=int(N * 0.04), replace=False)
    df.loc[idx, col] = np.nan

# --- Salva ---
df.to_csv(ARQUIVO, index=False)

print(f"Dataset gerado: {ARQUIVO}")
print(f"  Linhas    : {len(df)}")
print(f"  Colunas   : {df.shape[1]}")
print(f"  Churn     : {df['cancelou'].mean():.1%}  ({df['cancelou'].sum()} clientes)")
print(f"  Nulos     : score_credito={df['score_credito'].isna().sum()}, "
      f"variacao_gasto_pct={df['variacao_gasto_pct'].isna().sum()}, "
      f"canal_aquisicao={df['canal_aquisicao'].isna().sum()}")
print(f"\nPrimeiras linhas:")
print(df.head(5).to_string(index=False))