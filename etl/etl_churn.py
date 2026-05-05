# ============================================================
# ETL — PIPELINE COMPLETO: CSV → SQLite (Modelagem Dimensional)
# Projeto: Churn Prediction | Hackathon de Dados
# ============================================================
# O que este script faz:
#   E — Extract:   Lê o CSV do Kaggle e valida os dados
#   T — Transform: Limpa, enriquece e cria as dimensões
#   L — Load:      Cria o banco SQLite com modelo Star Schema
#
# Modelagem dimensional (Star Schema):
#
#   dim_cliente ──┐
#   dim_pais    ──┤
#   dim_produto ──┼──► fato_churn ◄── (tabela central de análise)
#   dim_risco   ──┘
#
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# ── Configuração de logs ──────────────────────────────────────
# Registramos cada etapa do ETL com timestamp para rastreabilidade.
# Em produção, logs são essenciais para monitorar falhas e auditar
# a execução do pipeline.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),                          # exibe no terminal
        logging.FileHandler('etl_churn.log', mode='w'),  # salva em arquivo
    ]
)
log = logging.getLogger(__name__)

# ── Configurações do pipeline ─────────────────────────────────
CSV_PATH = '/workspaces/hackathon-churn-prediction/dados/Bank_Customer_Churn_Prediction.csv'
DB_PATH  = 'churn_dw.db'   

# ════════════════════════════════════════════════════════════════
# FASE 1 — EXTRACT
# Lemos o CSV e realizamos validações básicas de qualidade antes
# de qualquer transformação. Se os dados estiverem corrompidos,
# falhamos aqui em vez de propagar erros silenciosos.
# ════════════════════════════════════════════════════════════════
def extract(path: str) -> pd.DataFrame:
    log.info("=" * 55)
    log.info("FASE 1 — EXTRACT")
    log.info("=" * 55)
    log.info(f"Lendo arquivo: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Certifique-se de que o CSV do Kaggle está na mesma pasta."
        )

    df = pd.read_csv(path)
    log.info(f"Arquivo lido com sucesso: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    # ── Validação de schema ──────────────────────────────────
    # Verificamos se todas as colunas esperadas estão presentes.
    # Isso protege o pipeline contra versões diferentes do dataset.
    colunas_esperadas = {
        'customer_id', 'credit_score', 'country', 'gender',
        'age', 'tenure', 'balance', 'products_number',
        'credit_card', 'active_member', 'estimated_salary', 'churn'
    }
    colunas_faltando = colunas_esperadas - set(df.columns)
    if colunas_faltando:
        raise ValueError(f"Colunas faltando no CSV: {colunas_faltando}")
    log.info("✅ Schema validado — todas as colunas esperadas presentes")

    # ── Validação de nulos ───────────────────────────────────
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        log.warning(f"⚠️  Valores nulos detectados:\n{nulos[nulos > 0]}")
    else:
        log.info("✅ Sem valores nulos")

    # ── Validação de duplicatas ──────────────────────────────
    n_dup = df.duplicated(subset='customer_id').sum()
    if n_dup > 0:
        log.warning(f"⚠️  {n_dup} customer_ids duplicados detectados")
    else:
        log.info("✅ Sem customer_ids duplicados")

    # ── Validação de domínios ────────────────────────────────
    # Verifica se os valores estão dentro dos domínios esperados
    assert df['churn'].isin([0, 1]).all(), "churn deve ser 0 ou 1"
    assert df['credit_card'].isin([0, 1]).all(), "credit_card deve ser 0 ou 1"
    assert df['active_member'].isin([0, 1]).all(), "active_member deve ser 0 ou 1"
    assert df['products_number'].between(1, 4).all(), "products_number deve ser 1–4"
    assert df['age'].between(18, 100).all(), "age fora do range esperado"
    log.info("✅ Validação de domínios concluída")

    log.info(f"Taxa de churn na origem: {df['churn'].mean():.1%}")
    return df


# ════════════════════════════════════════════════════════════════
# FASE 2 — TRANSFORM
# Limpamos, padronizamos e enriquecemos os dados para criar as
# tabelas do modelo dimensional. Cada função cria uma dimensão
# ou a tabela fato separadamente, facilitando manutenção e testes.
# ════════════════════════════════════════════════════════════════
def transform(df: pd.DataFrame) -> dict:
    log.info("=" * 55)
    log.info("FASE 2 — TRANSFORM")
    log.info("=" * 55)

    # ── 2.1 Limpeza geral ────────────────────────────────────
    df = df.copy()

    # Padroniza strings: remove espaços e aplica Title Case
    df['country'] = df['country'].str.strip().str.title()
    df['gender']  = df['gender'].str.strip().str.title()

    # Garante tipos corretos para evitar problemas de inserção no SQLite
    df['customer_id']      = df['customer_id'].astype(int)
    df['credit_score']     = df['credit_score'].astype(int)
    df['age']              = df['age'].astype(int)
    df['tenure']           = df['tenure'].astype(int)
    df['balance']          = df['balance'].astype(float).round(2)
    df['products_number']  = df['products_number'].astype(int)
    df['credit_card']      = df['credit_card'].astype(int)
    df['active_member']    = df['active_member'].astype(int)
    df['estimated_salary'] = df['estimated_salary'].astype(float).round(2)
    df['churn']            = df['churn'].astype(int)

    log.info("✅ Limpeza e padronização de tipos concluída")

    # ── 2.2 Feature engineering ──────────────────────────────
    # Criamos variáveis derivadas com base nos insights do EDA.
    # Essas colunas enriquecem o Data Warehouse e suportam análises
    # mais sofisticadas no Power BI e nas queries SQL.

    # Faixa etária — detectamos no EDA que idade é mais informativa
    # como categoria do que como valor contínuo para análise de churn
    df['faixa_etaria'] = pd.cut(
        df['age'],
        bins=[17, 30, 40, 50, 60, 100],
        labels=['18-30', '31-40', '41-50', '51-60', '61+']
    ).astype(str)

    # Flag de saldo — EDA revelou distribuição bimodal no balance:
    # clientes com saldo zero têm comportamento distinto dos demais
    df['tem_balance'] = (df['balance'] > 0).astype(int)

    # Faixa de score de crédito — categorização padrão do mercado financeiro
    df['faixa_credito'] = pd.cut(
        df['credit_score'],
        bins=[299, 499, 599, 699, 799, 1000],
        labels=['Muito Baixo', 'Baixo', 'Regular', 'Bom', 'Excelente']
    ).astype(str)

    # Faixa de relacionamento com o banco
    df['faixa_tenure'] = pd.cut(
        df['tenure'],
        bins=[-1, 2, 5, 7, 10],
        labels=['Novo (0-2)', 'Em desenvolvimento (3-5)',
                'Consolidado (6-7)', 'Fiel (8-10)']
    ).astype(str)

    # Score de risco financeiro simples (heurística do EDA)
    # Combina os 4 fatores de maior impacto identificados na análise
    df['score_risco_heuristico'] = (
        (df['active_member'] == 0).astype(int) * 30 +   # inativo = +30 pts risco
        (df['products_number'] >= 3).astype(int) * 25 + # muitos produtos = +25
        (df['age'] > 45).astype(int) * 20 +              # mais velho = +20
        (df['balance'] > 0).astype(int) * 15 +           # tem saldo = +15
        (df['credit_score'] < 600).astype(int) * 10      # score baixo = +10
    )
    # Normaliza para escala 0-100
    df['score_risco_heuristico'] = (
        df['score_risco_heuristico'] / df['score_risco_heuristico'].max() * 100
    ).round(1)

    log.info("✅ Feature engineering concluído (faixa_etaria, faixa_credito, faixa_tenure, score_risco)")

    # ── 2.3 DIM_PAIS ─────────────────────────────────────────
    # Dimensão dos países — contém atributos geográficos.
    # Em um DW real, teria mais atributos (continente, moeda, etc.)
    dim_pais = (
        df[['country']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dim_pais.insert(0, 'pais_id', range(1, len(dim_pais) + 1))
    dim_pais.columns = ['pais_id', 'pais_nome']
    # Adicionamos um atributo de região (conhecimento de domínio)
    regioes = {'France': 'Europa Ocidental',
               'Germany': 'Europa Central',
               'Spain': 'Europa do Sul'}
    dim_pais['regiao'] = dim_pais['pais_nome'].map(regioes)
    log.info(f"✅ dim_pais criada: {len(dim_pais)} registros")

    # ── 2.4 DIM_PRODUTO ──────────────────────────────────────
    # Dimensão dos produtos — descreve os segmentos de contratação.
    # Criada manualmente com base no conhecimento de negócio:
    # mais produtos = cliente mais "preso" ao banco, mas EDA mostrou
    # que 3+ produtos correlacionam com churn alto (surpreendente).
    dim_produto = pd.DataFrame({
        'produto_id':    [1, 2, 3, 4],
        'qtd_produtos':  [1, 2, 3, 4],
        'perfil':        ['Básico', 'Intermediário', 'Avançado', 'Premium'],
        'descricao':     [
            'Cliente com apenas 1 produto contratado',
            'Cliente com 2 produtos — perfil mais comum',
            'Cliente com 3 produtos — alto risco de churn detectado no EDA',
            'Cliente com 4 produtos — risco de churn muito elevado',
        ],
        'flag_alto_risco_churn': [0, 0, 1, 1]
    })
    log.info(f"✅ dim_produto criada: {len(dim_produto)} registros")

    # ── 2.5 DIM_CLIENTE ──────────────────────────────────────
    # Dimensão do cliente — atributos descritivos que não mudam
    # frequentemente (slowly changing dimension tipo 1).
    dim_cliente = df[[
        'customer_id', 'gender', 'age', 'faixa_etaria',
        'credit_score', 'faixa_credito', 'estimated_salary',
        'tenure', 'faixa_tenure', 'credit_card'
    ]].copy()
    dim_cliente.columns = [
        'cliente_id', 'genero', 'idade', 'faixa_etaria',
        'score_credito', 'faixa_credito', 'salario_estimado',
        'tempo_relacionamento', 'faixa_tenure', 'tem_cartao_credito'
    ]
    # Metadado de quando o registro foi carregado no DW
    dim_cliente['data_carga'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"✅ dim_cliente criada: {len(dim_cliente)} registros")

    # ── 2.6 DIM_RISCO ────────────────────────────────────────
    # Dimensão de risco — classifica os clientes em faixas de risco
    # com base no score heurístico calculado no feature engineering.
    # Esta dimensão conecta a análise exploratória ao Data Warehouse.
    def classificar_risco(score):
        if score >= 70:   return 'Crítico'
        elif score >= 50: return 'Alto'
        elif score >= 30: return 'Médio'
        else:             return 'Baixo'

    dim_risco_base = pd.DataFrame({
        'risco_id':    [1, 2, 3, 4],
        'faixa_risco': ['Baixo', 'Médio', 'Alto', 'Crítico'],
        'score_min':   [0,  30, 50, 70],
        'score_max':   [29, 49, 69, 100],
        'descricao':   [
            'Baixo risco de churn — cliente estável',
            'Risco moderado — monitorar comportamento',
            'Alto risco — acionar equipe de retenção',
            'Risco crítico — intervenção imediata necessária',
        ],
        'cor_alerta':  ['#1D9E75', '#F5A623', '#E8593C', '#8B0000']
    })

    # Criamos um mapeamento para associar cada cliente ao risco_id na fato
    df['faixa_risco_str'] = df['score_risco_heuristico'].apply(classificar_risco)
    risco_map = dict(zip(dim_risco_base['faixa_risco'], dim_risco_base['risco_id']))
    df['risco_id'] = df['faixa_risco_str'].map(risco_map)

    log.info(f"✅ dim_risco criada: {len(dim_risco_base)} registros")
    log.info(f"   Distribuição: {df['faixa_risco_str'].value_counts().to_dict()}")

    # ── 2.7 FATO_CHURN ───────────────────────────────────────
    # Tabela fato — é o coração do Star Schema. Contém as medidas
    # quantitativas (fatos) e as chaves estrangeiras para as dimensões.
    # Cada linha representa um cliente em um snapshot no tempo.

    # Cria mapeamento de country → pais_id
    pais_map = dict(zip(dim_pais['pais_nome'], dim_pais['pais_id']))
    df['pais_id'] = df['country'].map(pais_map)

    fato_churn = df[[
        'customer_id',       # FK → dim_cliente
        'pais_id',           # FK → dim_pais
        'products_number',   # FK → dim_produto (via qtd_produtos)
        'risco_id',          # FK → dim_risco
        'balance',           # FATO: saldo em conta
        'score_risco_heuristico',  # FATO: score de risco calculado
        'active_member',     # FATO: flag de atividade
        'tem_balance',       # FATO: flag derivada do EDA
        'churn',             # FATO: variável alvo (resultado)
    ]].copy()

    fato_churn.columns = [
        'cliente_id', 'pais_id', 'qtd_produtos', 'risco_id',
        'balance', 'score_risco', 'membro_ativo',
        'tem_balance', 'churn'
    ]
    # Chave surrogate da fato (boa prática em DW)
    fato_churn.insert(0, 'fato_id', range(1, len(fato_churn) + 1))
    fato_churn['data_referencia'] = datetime.now().strftime('%Y-%m-%d')

    log.info(f"✅ fato_churn criada: {len(fato_churn)} registros")

    return {
        'dim_pais':     dim_pais,
        'dim_produto':  dim_produto,
        'dim_cliente':  dim_cliente,
        'dim_risco':    dim_risco_base,
        'fato_churn':   fato_churn,
    }


# ════════════════════════════════════════════════════════════════
# FASE 3 — LOAD
# Criamos o banco SQLite, definimos as tabelas com DDL explícito
# (incluindo chaves primárias, estrangeiras e índices) e
# carregamos os dados transformados.
# ════════════════════════════════════════════════════════════════
def load(tabelas: dict, db_path: str) -> None:
    log.info("=" * 55)
    log.info("FASE 3 — LOAD")
    log.info("=" * 55)

    # Remove banco anterior para garantir idempotência:
    # rodar o ETL duas vezes deve produzir o mesmo resultado
    if os.path.exists(db_path):
        os.remove(db_path)
        log.info(f"Banco anterior removido: {db_path}")

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Ativa suporte a chaves estrangeiras no SQLite
    # (desabilitado por padrão — boa prática ativar sempre)
    cur.execute("PRAGMA foreign_keys = ON;")

    # ── DDL: Criação das tabelas ─────────────────────────────
    # Usamos DDL explícito (em vez de to_sql direto) para controlar
    # tipos, constraints e índices — produzindo um schema profissional.

    log.info("Criando schema do banco de dados...")

    cur.executescript("""
    -- ────────────────────────────────────────────────────────
    -- DIMENSÕES
    -- ────────────────────────────────────────────────────────

    CREATE TABLE IF NOT EXISTS dim_pais (
        pais_id     INTEGER PRIMARY KEY,
        pais_nome   TEXT    NOT NULL UNIQUE,
        regiao      TEXT
    );

    CREATE TABLE IF NOT EXISTS dim_produto (
        produto_id              INTEGER PRIMARY KEY,
        qtd_produtos            INTEGER NOT NULL UNIQUE,
        perfil                  TEXT    NOT NULL,
        descricao               TEXT,
        flag_alto_risco_churn   INTEGER NOT NULL DEFAULT 0
            CHECK (flag_alto_risco_churn IN (0, 1))
    );

    CREATE TABLE IF NOT EXISTS dim_risco (
        risco_id    INTEGER PRIMARY KEY,
        faixa_risco TEXT    NOT NULL UNIQUE,
        score_min   INTEGER NOT NULL,
        score_max   INTEGER NOT NULL,
        descricao   TEXT,
        cor_alerta  TEXT
    );

    CREATE TABLE IF NOT EXISTS dim_cliente (
        cliente_id          INTEGER PRIMARY KEY,
        genero              TEXT    NOT NULL,
        idade               INTEGER NOT NULL CHECK (idade BETWEEN 18 AND 100),
        faixa_etaria        TEXT,
        score_credito       INTEGER NOT NULL CHECK (score_credito BETWEEN 300 AND 850),
        faixa_credito       TEXT,
        salario_estimado    REAL,
        tempo_relacionamento INTEGER,
        faixa_tenure        TEXT,
        tem_cartao_credito  INTEGER NOT NULL DEFAULT 0
            CHECK (tem_cartao_credito IN (0, 1)),
        data_carga          TEXT
    );

    -- ────────────────────────────────────────────────────────
    -- TABELA FATO (coração do Star Schema)
    -- ────────────────────────────────────────────────────────

    CREATE TABLE IF NOT EXISTS fato_churn (
        fato_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        cliente_id       INTEGER NOT NULL REFERENCES dim_cliente(cliente_id),
        pais_id          INTEGER NOT NULL REFERENCES dim_pais(pais_id),
        qtd_produtos     INTEGER NOT NULL REFERENCES dim_produto(qtd_produtos),
        risco_id         INTEGER NOT NULL REFERENCES dim_risco(risco_id),
        balance          REAL    NOT NULL,
        score_risco      REAL    NOT NULL,
        membro_ativo     INTEGER NOT NULL CHECK (membro_ativo IN (0, 1)),
        tem_balance      INTEGER NOT NULL CHECK (tem_balance IN (0, 1)),
        churn            INTEGER NOT NULL CHECK (churn IN (0, 1)),
        data_referencia  TEXT    NOT NULL
    );

    -- ────────────────────────────────────────────────────────
    -- ÍNDICES: aceleram as queries mais comuns no Power BI e SQL
    -- ────────────────────────────────────────────────────────
    CREATE INDEX IF NOT EXISTS idx_fato_churn      ON fato_churn(churn);
    CREATE INDEX IF NOT EXISTS idx_fato_cliente    ON fato_churn(cliente_id);
    CREATE INDEX IF NOT EXISTS idx_fato_pais       ON fato_churn(pais_id);
    CREATE INDEX IF NOT EXISTS idx_fato_risco      ON fato_churn(risco_id);
    CREATE INDEX IF NOT EXISTS idx_cliente_faixa   ON dim_cliente(faixa_etaria);
    CREATE INDEX IF NOT EXISTS idx_cliente_credito ON dim_cliente(faixa_credito);
    """)

    log.info("✅ Schema criado com sucesso")

    # ── Inserção dos dados ───────────────────────────────────
    # Carregamos na ordem correta: dimensões primeiro, fato por último
    # (respeita as foreign key constraints)
    ordem_carga = ['dim_pais', 'dim_produto', 'dim_risco', 'dim_cliente', 'fato_churn']

    for nome_tabela in ordem_carga:
        df_tab = tabelas[nome_tabela]
        df_tab.to_sql(nome_tabela, conn, if_exists='append', index=False)
        log.info(f"  ✅ {nome_tabela}: {len(df_tab):,} registros inseridos")

    conn.commit()

    # ── Verificação pós-carga ────────────────────────────────
    log.info("\nVerificação pós-carga:")
    for tabela in ordem_carga:
        count = cur.execute(f"SELECT COUNT(*) FROM {tabela}").fetchone()[0]
        log.info(f"  {tabela:<20} → {count:,} registros")

    # ── Cria views analíticas ────────────────────────────────
   
    log.info("\nCriando views analíticas...")

    cur.executescript("""
    -- View 1: Visão completa do cliente com todas as dimensões
    CREATE VIEW IF NOT EXISTS vw_cliente_completo AS
    SELECT
        f.fato_id,
        f.cliente_id,
        c.genero,
        c.idade,
        c.faixa_etaria,
        c.score_credito,
        c.faixa_credito,
        c.tempo_relacionamento,
        c.faixa_tenure,
        c.tem_cartao_credito,
        p.pais_nome,
        p.regiao,
        pr.perfil           AS perfil_produto,
        pr.flag_alto_risco_churn,
        r.faixa_risco,
        r.cor_alerta,
        f.balance,
        f.tem_balance,
        f.score_risco,
        f.membro_ativo,
        f.churn,
        CASE f.churn
            WHEN 1 THEN 'Cancelou'
            ELSE 'Ativo'
        END                 AS status_cliente,
        f.data_referencia
    FROM fato_churn f
    JOIN dim_cliente  c  ON f.cliente_id    = c.cliente_id
    JOIN dim_pais     p  ON f.pais_id       = p.pais_id
    JOIN dim_produto  pr ON f.qtd_produtos  = pr.qtd_produtos
    JOIN dim_risco    r  ON f.risco_id      = r.risco_id;

    -- View 2: KPIs executivos por país
    CREATE VIEW IF NOT EXISTS vw_kpi_por_pais AS
    SELECT
        p.pais_nome,
        p.regiao,
        COUNT(*)                                    AS total_clientes,
        SUM(f.churn)                                AS total_churn,
        ROUND(AVG(f.churn) * 100, 2)                AS taxa_churn_pct,
        ROUND(AVG(c.score_credito), 1)              AS score_credito_medio,
        ROUND(AVG(f.balance), 2)                    AS balance_medio,
        ROUND(AVG(f.score_risco), 1)                AS risco_medio,
        SUM(CASE WHEN f.membro_ativo = 0 THEN 1 ELSE 0 END) AS clientes_inativos
    FROM fato_churn f
    JOIN dim_pais    p ON f.pais_id    = p.pais_id
    JOIN dim_cliente c ON f.cliente_id = c.cliente_id
    GROUP BY p.pais_nome, p.regiao;

    -- View 3: KPIs por faixa etária
    CREATE VIEW IF NOT EXISTS vw_kpi_por_faixa_etaria AS
    SELECT
        c.faixa_etaria,
        COUNT(*)                             AS total_clientes,
        SUM(f.churn)                         AS total_churn,
        ROUND(AVG(f.churn) * 100, 2)         AS taxa_churn_pct,
        ROUND(AVG(c.score_credito), 1)       AS score_credito_medio,
        ROUND(AVG(f.balance), 2)             AS balance_medio,
        ROUND(AVG(f.score_risco), 1)         AS risco_medio
    FROM fato_churn f
    JOIN dim_cliente c ON f.cliente_id = c.cliente_id
    GROUP BY c.faixa_etaria
    ORDER BY c.faixa_etaria;

    -- View 4: Clientes de alto risco (para dashboard de alertas)
    CREATE VIEW IF NOT EXISTS vw_alto_risco AS
    SELECT
        f.cliente_id,
        c.genero,
        c.idade,
        c.faixa_etaria,
        p.pais_nome,
        f.balance,
        f.score_risco,
        r.faixa_risco,
        f.membro_ativo,
        f.churn
    FROM fato_churn f
    JOIN dim_cliente c ON f.cliente_id = c.cliente_id
    JOIN dim_pais    p ON f.pais_id    = p.pais_id
    JOIN dim_risco   r ON f.risco_id   = r.risco_id
    WHERE r.faixa_risco IN ('Alto', 'Crítico')
    ORDER BY f.score_risco DESC;
    """)
    conn.commit()
    log.info("✅ 4 views analíticas criadas")

    conn.close()
    tamanho_kb = os.path.getsize(db_path) / 1024
    log.info(f"\n✅ Banco de dados criado: {db_path} ({tamanho_kb:.1f} KB)")


# ════════════════════════════════════════════════════════════════
# EXECUÇÃO DO PIPELINE
# ════════════════════════════════════════════════════════════════
def main():
    inicio = datetime.now()
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   ETL PIPELINE — CHURN PREDICTION DATA WAREHOUSE    ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info(f"Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # E — Extract
        df_raw = extract(CSV_PATH)

        # T — Transform
        tabelas = transform(df_raw)

        # L — Load
        load(tabelas, DB_PATH)

        # Resumo final
        fim = datetime.now()
        duracao = (fim - inicio).total_seconds()

        log.info("")
        log.info("╔══════════════════════════════════════════════════════╗")
        log.info("║                  PIPELINE CONCLUÍDO                 ║")
        log.info("╠══════════════════════════════════════════════════════╣")
        log.info(f"║  Banco criado : {DB_PATH:<38}║")
        log.info(f"║  Log salvo    : {'etl_churn.log':<38}║")
        log.info(f"║  Duração      : {duracao:.2f}s{'':<35}║")
        log.info("╠══════════════════════════════════════════════════════╣")
        log.info("║  Tabelas:                                            ║")
        log.info("║    dim_pais | dim_produto | dim_risco | dim_cliente  ║")
        log.info("║    fato_churn                                        ║")
        log.info("║  Views:                                              ║")
        log.info("║    vw_cliente_completo  | vw_kpi_por_pais            ║")
        log.info("║    vw_kpi_por_faixa_etaria | vw_alto_risco           ║")
        log.info("╚══════════════════════════════════════════════════════╝")

    except FileNotFoundError as e:
        log.error(f"❌ Arquivo não encontrado: {e}")
        raise
    except Exception as e:
        log.error(f"❌ Erro no pipeline: {e}")
        raise


if __name__ == '__main__':
    main()