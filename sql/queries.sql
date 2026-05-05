-- ============================================================
-- QUERIES SQL — CHURN PREDICTION 
-- Banco: churn_dw.db (SQLite)
-- Gerado pelo ETL: etl_churn.py
-- ============================================================
-- Como executar:
--   sqlite3 churn_dw.db < queries.sql
--   Ou abrir o .db no DB Browser for SQLite e rodar por seção
-- ============================================================


-- ============================================================
-- 1. INSPEÇÃO INICIAL DO BANCO
-- Verifica se o ETL foi executado corretamente e o banco
-- está íntegro antes de iniciar as análises.
-- ============================================================

-- 1.1 Contagem de registros por tabela
SELECT 'dim_cliente'  AS tabela, COUNT(*) AS registros FROM dim_cliente
UNION ALL
SELECT 'dim_pais',     COUNT(*) FROM dim_pais
UNION ALL
SELECT 'dim_produto',  COUNT(*) FROM dim_produto
UNION ALL
SELECT 'dim_risco',    COUNT(*) FROM dim_risco
UNION ALL
SELECT 'fato_churn',   COUNT(*) FROM fato_churn;

-- 1.2 Amostra da view completa (top 10)
SELECT *
FROM vw_cliente_completo
LIMIT 10;

-- 1.3 Verificação de integridade: clientes na fato sem dimensão
-- Resultado esperado: 0 registros (foreign keys garantem isso)
SELECT f.cliente_id
FROM fato_churn f
LEFT JOIN dim_cliente c ON f.cliente_id = c.cliente_id
WHERE c.cliente_id IS NULL;


-- ============================================================
-- 2. ANÁLISE GERAL DE CHURN
-- Visão macro do problema: taxa, volume e distribuição.
-- ============================================================

-- 2.1 Taxa geral de churn
SELECT
    COUNT(*)                            AS total_clientes,
    SUM(churn)                          AS total_churned,
    COUNT(*) - SUM(churn)               AS total_ativos,
    ROUND(AVG(churn) * 100, 2)          AS taxa_churn_pct,
    ROUND((1 - AVG(churn)) * 100, 2)    AS taxa_retencao_pct
FROM fato_churn;

-- 2.2 Distribuição por faixa de risco
-- Mostra quantos clientes estão em cada nível de alerta
SELECT
    r.faixa_risco,
    r.descricao,
    COUNT(*)                            AS total_clientes,
    SUM(f.churn)                        AS churned,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct,
    r.cor_alerta
FROM fato_churn f
JOIN dim_risco r ON f.risco_id = r.risco_id
GROUP BY r.faixa_risco, r.descricao, r.cor_alerta
ORDER BY r.risco_id DESC;


-- ============================================================
-- 3. ANÁLISE POR SEGMENTO GEOGRÁFICO
-- Identifica diferenças de comportamento entre países.
-- Insight do EDA: Alemanha tem taxa de churn significativamente
-- maior que França e Espanha.
-- ============================================================

-- 3.1 KPIs completos por país
SELECT *
FROM vw_kpi_por_pais
ORDER BY taxa_churn_pct DESC;

-- 3.2 Churn por país e gênero (análise cruzada)
-- Verifica se o padrão de gênero é consistente entre países
SELECT
    p.pais_nome,
    c.genero,
    COUNT(*)                            AS total,
    SUM(f.churn)                        AS churned,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct
FROM fato_churn f
JOIN dim_cliente c ON f.cliente_id = c.cliente_id
JOIN dim_pais    p ON f.pais_id    = p.pais_id
GROUP BY p.pais_nome, c.genero
ORDER BY p.pais_nome, taxa_churn_pct DESC;

-- 3.3 Países acima da média de churn (filtragem condicional)
-- Útil para focar ações de retenção nas regiões mais críticas
SELECT
    pais_nome,
    taxa_churn_pct,
    total_clientes,
    ROUND(taxa_churn_pct - (SELECT AVG(churn)*100 FROM fato_churn), 2) AS desvio_da_media
FROM vw_kpi_por_pais
WHERE taxa_churn_pct > (SELECT AVG(churn) * 100 FROM fato_churn)
ORDER BY taxa_churn_pct DESC;


-- ============================================================
-- 4. ANÁLISE POR PERFIL DO CLIENTE
-- Segmentações demográficas que revelam padrões de churn.
-- ============================================================

-- 4.1 Churn por faixa etária (insight crítico do EDA)
SELECT *
FROM vw_kpi_por_faixa_etaria;

-- 4.2 Churn por faixa de score de crédito
SELECT
    c.faixa_credito,
    COUNT(*)                            AS total_clientes,
    SUM(f.churn)                        AS churned,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct,
    ROUND(AVG(c.score_credito), 0)      AS score_medio
FROM fato_churn f
JOIN dim_cliente c ON f.cliente_id = c.cliente_id
GROUP BY c.faixa_credito
ORDER BY score_medio;

-- 4.3 Churn por tempo de relacionamento com o banco
SELECT
    c.faixa_tenure,
    COUNT(*)                            AS total_clientes,
    SUM(f.churn)                        AS churned,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct,
    ROUND(AVG(c.tempo_relacionamento), 1) AS tenure_medio
FROM fato_churn f
JOIN dim_cliente c ON f.cliente_id = c.cliente_id
GROUP BY c.faixa_tenure
ORDER BY tenure_medio;

-- 4.4 Impacto do status de membro ativo no churn
-- Insight do EDA: membros inativos churnam ~2x mais
SELECT
    CASE f.membro_ativo WHEN 1 THEN 'Ativo' ELSE 'Inativo' END AS status_membro,
    COUNT(*)                            AS total_clientes,
    SUM(f.churn)                        AS churned,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct
FROM fato_churn f
GROUP BY f.membro_ativo
ORDER BY f.membro_ativo DESC;


-- ============================================================
-- 5. ANÁLISE POR PERFIL DE PRODUTO
-- O número de produtos é uma das features mais importantes
-- detectadas no EDA — com padrão contra-intuitivo.
-- ============================================================

-- 5.1 Churn por número de produtos
-- Insight: clientes com 3+ produtos têm churn elevado
SELECT
    pr.qtd_produtos,
    pr.perfil,
    pr.flag_alto_risco_churn,
    COUNT(*)                            AS total_clientes,
    SUM(f.churn)                        AS churned,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct
FROM fato_churn f
JOIN dim_produto pr ON f.qtd_produtos = pr.qtd_produtos
GROUP BY pr.qtd_produtos, pr.perfil, pr.flag_alto_risco_churn
ORDER BY pr.qtd_produtos;

-- 5.2 Produtos × saldo: combinação de risco
SELECT
    pr.perfil,
    CASE f.tem_balance WHEN 1 THEN 'Com saldo' ELSE 'Sem saldo' END AS situacao_saldo,
    COUNT(*)                            AS total,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct,
    ROUND(AVG(f.balance), 2)            AS balance_medio
FROM fato_churn f
JOIN dim_produto pr ON f.qtd_produtos = pr.qtd_produtos
GROUP BY pr.perfil, f.tem_balance
ORDER BY pr.qtd_produtos, f.tem_balance DESC;


-- ============================================================
-- 6. ANÁLISE FINANCEIRA
-- Impacto do saldo e score de risco no comportamento de churn.
-- ============================================================

-- 6.1 Clientes com saldo vs sem saldo
SELECT
    CASE tem_balance WHEN 1 THEN 'Com saldo' ELSE 'Sem saldo' END AS situacao,
    COUNT(*)                            AS total_clientes,
    SUM(churn)                          AS churned,
    ROUND(AVG(churn) * 100, 2)          AS taxa_churn_pct,
    ROUND(AVG(balance), 2)              AS balance_medio,
    ROUND(MIN(balance), 2)              AS balance_min,
    ROUND(MAX(balance), 2)              AS balance_max
FROM fato_churn
GROUP BY tem_balance;

-- 6.2 Estatísticas de saldo por status de churn
SELECT
    CASE churn WHEN 1 THEN 'Churned' ELSE 'Ativo' END AS status,
    COUNT(*)                            AS total,
    ROUND(AVG(balance), 2)              AS balance_medio,
    ROUND(MIN(balance), 2)              AS balance_min,
    ROUND(MAX(balance), 2)              AS balance_max,
    ROUND(
        (MAX(balance) - MIN(balance)) /
        NULLIF((SELECT AVG(balance) FROM fato_churn), 0) * 100
    , 1)                                AS amplitude_relativa_pct
FROM fato_churn
GROUP BY churn;

-- 6.3 Score de risco médio por segmento (faixa etária × país)
-- Identifica onde concentrar ações preventivas
SELECT
    p.pais_nome,
    c.faixa_etaria,
    COUNT(*)                            AS total_clientes,
    ROUND(AVG(f.score_risco), 1)        AS score_risco_medio,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct
FROM fato_churn f
JOIN dim_cliente c ON f.cliente_id = c.cliente_id
JOIN dim_pais    p ON f.pais_id    = p.pais_id
GROUP BY p.pais_nome, c.faixa_etaria
ORDER BY score_risco_medio DESC
LIMIT 15;


-- ============================================================
-- 7. LIMPEZA DE DADOS — TRATAMENTO DE NULOS
-- Demonstra boas práticas de SQL: identificação e tratamento
-- de valores ausentes ou inconsistentes.
-- ============================================================

-- 7.1 Verificação de nulos nas colunas da fato
SELECT
    SUM(CASE WHEN cliente_id   IS NULL THEN 1 ELSE 0 END) AS nulos_cliente_id,
    SUM(CASE WHEN pais_id      IS NULL THEN 1 ELSE 0 END) AS nulos_pais_id,
    SUM(CASE WHEN balance      IS NULL THEN 1 ELSE 0 END) AS nulos_balance,
    SUM(CASE WHEN score_risco  IS NULL THEN 1 ELSE 0 END) AS nulos_score_risco,
    SUM(CASE WHEN churn        IS NULL THEN 1 ELSE 0 END) AS nulos_churn
FROM fato_churn;

-- 7.2 Verificação de valores fora do domínio esperado
-- Constraints do banco previnem isso, mas é boa prática verificar
SELECT
    SUM(CASE WHEN churn NOT IN (0,1)           THEN 1 ELSE 0 END) AS churn_invalido,
    SUM(CASE WHEN membro_ativo NOT IN (0,1)    THEN 1 ELSE 0 END) AS ativo_invalido,
    SUM(CASE WHEN balance < 0                  THEN 1 ELSE 0 END) AS balance_negativo,
    SUM(CASE WHEN score_risco < 0              THEN 1 ELSE 0 END) AS score_negativo
FROM fato_churn;

-- 7.3 Tratamento de nulos na dim_cliente (COALESCE)
-- Demonstra como lidar com campos opcionais de forma segura
SELECT
    cliente_id,
    COALESCE(faixa_etaria,  'Não informado') AS faixa_etaria,
    COALESCE(faixa_credito, 'Não informado') AS faixa_credito,
    COALESCE(faixa_tenure,  'Não informado') AS faixa_tenure
FROM dim_cliente
WHERE faixa_etaria IS NULL
   OR faixa_credito IS NULL
   OR faixa_tenure IS NULL
LIMIT 10;


-- ============================================================
-- 8. QUERIES AVANÇADAS
-- Análises mais sofisticadas para o dashboard executivo
-- e para suporte às decisões de retenção.
-- ============================================================

-- 8.1 Top 20 clientes de maior risco (alerta para retenção)
SELECT
    cliente_id,
    genero,
    faixa_etaria,
    pais_nome,
    ROUND(balance, 2)                   AS saldo,
    score_risco,
    faixa_risco,
    CASE membro_ativo WHEN 0 THEN 'Inativo' ELSE 'Ativo' END AS status_membro,
    churn
FROM vw_alto_risco
LIMIT 20;

-- 8.2 Perfil médio: cliente que churnou vs que ficou
-- Útil para descrever o "avatar" de cada grupo na apresentação
SELECT
    CASE churn WHEN 1 THEN 'Churned' ELSE 'Ativo' END   AS perfil,
    COUNT(*)                                             AS total,
    ROUND(AVG(c.idade), 1)                               AS idade_media,
    ROUND(AVG(c.score_credito), 0)                       AS score_credito_medio,
    ROUND(AVG(c.tempo_relacionamento), 1)                AS tenure_medio,
    ROUND(AVG(f.balance), 2)                             AS saldo_medio,
    ROUND(AVG(f.qtd_produtos), 2)                        AS produtos_medio,
    ROUND(AVG(f.score_risco), 1)                         AS score_risco_medio,
    ROUND(AVG(f.membro_ativo) * 100, 1)                  AS pct_membros_ativos
FROM fato_churn f
JOIN dim_cliente c ON f.cliente_id = c.cliente_id
GROUP BY churn;

-- 8.3 Ranking de combinações de risco (faixa etária × produtos)
-- Replica o heatmap multivariado do EDA em SQL
SELECT
    c.faixa_etaria,
    pr.perfil                           AS perfil_produto,
    COUNT(*)                            AS total_clientes,
    SUM(f.churn)                        AS churned,
    ROUND(AVG(f.churn) * 100, 2)        AS taxa_churn_pct
FROM fato_churn f
JOIN dim_cliente c  ON f.cliente_id   = c.cliente_id
JOIN dim_produto pr ON f.qtd_produtos = pr.qtd_produtos
GROUP BY c.faixa_etaria, pr.perfil
ORDER BY taxa_churn_pct DESC;

-- 8.4 Clientes inativos com alto saldo e sem churn ainda
-- Lista de oportunidades de retenção proativa
SELECT
    f.cliente_id,
    c.faixa_etaria,
    p.pais_nome,
    ROUND(f.balance, 2)                 AS saldo,
    c.tempo_relacionamento              AS tenure,
    f.score_risco,
    r.faixa_risco
FROM fato_churn f
JOIN dim_cliente c ON f.cliente_id = c.cliente_id
JOIN dim_pais    p ON f.pais_id    = p.pais_id
JOIN dim_risco   r ON f.risco_id   = r.risco_id
WHERE f.membro_ativo = 0        -- membro inativo
  AND f.tem_balance  = 1        -- tem saldo no banco
  AND f.churn        = 0        -- ainda não cancelou
  AND r.faixa_risco IN ('Alto', 'Crítico')
ORDER BY f.balance DESC
LIMIT 20;

-- 8.5 Receita em risco (estimativa de impacto financeiro do churn)
-- Soma o saldo dos clientes que já churinaram (perda realizada)
-- e dos que estão em alto risco (perda potencial)
SELECT
    'Perda realizada (já churinaram)' AS categoria,
    COUNT(*)                          AS qtd_clientes,
    ROUND(SUM(balance), 2)            AS saldo_total_em_risco
FROM fato_churn
WHERE churn = 1
UNION ALL
SELECT
    'Perda potencial (alto/crítico, ainda ativos)',
    COUNT(*),
    ROUND(SUM(f.balance), 2)
FROM fato_churn f
JOIN dim_risco r ON f.risco_id = r.risco_id
WHERE f.churn = 0
  AND r.faixa_risco IN ('Alto', 'Crítico');

-- 8.6 Evolução simulada: distribuição de risco por faixa etária
-- Para uso no storytelling da apresentação executiva
SELECT
    c.faixa_etaria,
    SUM(CASE WHEN r.faixa_risco = 'Baixo'   THEN 1 ELSE 0 END) AS risco_baixo,
    SUM(CASE WHEN r.faixa_risco = 'Médio'   THEN 1 ELSE 0 END) AS risco_medio,
    SUM(CASE WHEN r.faixa_risco = 'Alto'    THEN 1 ELSE 0 END) AS risco_alto,
    SUM(CASE WHEN r.faixa_risco = 'Crítico' THEN 1 ELSE 0 END) AS risco_critico,
    COUNT(*)                                                    AS total,
    ROUND(AVG(f.churn) * 100, 2)                                AS taxa_churn_pct
FROM fato_churn f
JOIN dim_cliente c ON f.cliente_id = c.cliente_id
JOIN dim_risco   r ON f.risco_id   = r.risco_id
GROUP BY c.faixa_etaria
ORDER BY c.faixa_etaria;


-- ============================================================
-- 9. VIEWS ÚTEIS PARA O POWER BI
-- Consultas prontas para conectar diretamente no Power BI
-- via ODBC ou importação do arquivo .db.
-- ============================================================

-- 9.1 Tabela flat para o Power BI (substitui o CSV)
SELECT
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
    f.qtd_produtos,
    pr.perfil                           AS perfil_produto,
    pr.flag_alto_risco_churn,
    f.balance,
    f.tem_balance,
    f.membro_ativo,
    f.score_risco,
    r.faixa_risco,
    r.cor_alerta,
    f.churn,
    CASE f.churn WHEN 1 THEN 'Churned' ELSE 'Ativo' END AS status_cliente,
    f.data_referencia
FROM fato_churn f
JOIN dim_cliente c  ON f.cliente_id   = c.cliente_id
JOIN dim_pais    p  ON f.pais_id      = p.pais_id
JOIN dim_produto pr ON f.qtd_produtos = pr.qtd_produtos
JOIN dim_risco   r  ON f.risco_id     = r.risco_id;

