# Insights sobre Churn de Clientes

## Visão Geral

Este projeto analisa o comportamento de churn (evasão) de clientes utilizando técnicas de Machine Learning e explicabilidade com SHAP.

O objetivo é identificar clientes com alta probabilidade de churn e gerar insights de negócios para apoiar estratégias de retenção.

Dataset utilizado:

- Bank Customer Churn Prediction (Kaggle)
- 10.000 clientes
- 12 variáveis
- Taxa de churn de aproximadamente 20%

---

# Principais Insights de Negócios

## 1. A idade é o fator mais importante para o churn

A análise SHAP demonstrou que a idade foi a variável mais influente na previsão de churn.

Principais descobertas:

- Clientes mais velhos apresentaram maior probabilidade de churn
- Clientes mais jovens demonstraram maiores taxas de retenção

Impacto nos negócios:

Clientes acima de 40–50 anos podem exigir estratégias de retenção proativas.

---

## 2. Clientes inativos apresentam o maior risco de churn

A variável `active_member` mostrou forte influência na previsão de churn.

Principais descobertas:

- Clientes inativos tiveram uma probabilidade significativamente maior de churn
- Clientes ativos apresentaram menor probabilidade de churn

Ações recomendadas:

- Campanhas de engajamento
- Comunicação personalizada
- Programas de fidelidade
- Acompanhamento de Customer Success

---

## 3. O número de produtos influencia a retenção

A variável `products_number` revelou padrões comportamentais importantes.

Principais descobertas:

- Clientes com apenas um produto apresentaram maior concentração de churn
- Clientes com múltiplos produtos demonstraram melhor retenção

Oportunidade de negócios:

Estratégias de cross-selling podem reduzir o risco de churn.

---

## 4. Clientes com saldo alto representam risco financeiro

A variável `balance` demonstrou influência relevante na previsão de churn.

Principais descobertas:

- Clientes com saldos mais altos podem gerar maior impacto financeiro em caso de churn
- A exposição financeira aumenta significativamente entre clientes de alto valor

Impacto nos negócios:

Os esforços de retenção devem priorizar clientes de alto valor.

---

## 5. A posse de cartão de crédito teve baixa influência

A variável `credit_card` mostrou impacto mínimo na previsão de churn.

Insight:

Ter um cartão de crédito, por si só, não é um forte indicador de retenção.

---

# Explicabilidade SHAP

A análise SHAP permitiu a interpretação das decisões do modelo e a identificação das variáveis mais importantes que afetam a previsão de churn.

Principais Variáveis:

1. Idade
2. Número de Produtos
3. Membro Ativo
4. Saldo
5. País

A camada de explicabilidade melhora a transparência do modelo e apoia a tomada de decisões de negócios.

---

# Desempenho do Modelo

| Métrica | Valor |
|---|---|
| **AUC-ROC** (Gradient Boosting) | **0.8756** |
| Recall da classe Churn (threshold ajustado) | **~68%** |
| F1-Score Churn | **0.61** |
| Acurácia geral | **87%** |
| Validação cruzada (5-fold) | **0.8625 ± 0.0042** |

---

# Recomendações Estratégicas

## Clientes de Alta Prioridade

Focar as ações de retenção em:

- Clientes inativos
- Clientes acima de 50 anos
- Clientes com apenas um produto
- Clientes com saldo alto

---

## Ações Sugeridas

- Campanhas de retenção personalizadas
- Ofertas de multi-produtos
- Iniciativas de engajamento de clientes
- Monitoramento proativo de churn
- Segmentação de clientes de alto risco

---

# Conclusão

O projeto identificou com sucesso os principais impulsionadores do churn de clientes e demonstrou como o Machine Learning pode apoiar estratégias proativas de retenção.

A combinação da modelagem preditiva com a explicabilidade SHAP e a análise financeira permite melhores decisões de negócios e a priorização dos esforços de retenção.