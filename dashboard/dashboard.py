import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3, os, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
warnings.filterwarnings('ignore')

# ── Tema e configuração da página ────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence | Hackathon",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0a0e1a; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0a0e1a 0%, #0f1929 50%, #0a0e1a 100%); }
[data-testid="stSidebar"] { background: #0d1422 !important; border-right: 1px solid #1D9E7530; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid #1D9E7520;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1D9E75, #3B8BD4);
}
.metric-card:hover { border-color: #1D9E7550; transform: translateY(-2px); }
.metric-value { font-size: 2.4em; font-weight: 700; color: #1D9E75; line-height: 1; }
.metric-value.danger { color: #E8593C; }
.metric-value.info   { color: #3B8BD4; }
.metric-value.warn   { color: #F59E0B; }
.metric-label { font-size: 0.78em; color: #6b7280; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 8px; }
.metric-delta { font-size: 0.82em; color: #9ca3af; margin-top: 4px; }

.section-header {
    border-left: 4px solid #1D9E75;
    padding: 12px 20px;
    background: linear-gradient(90deg, #1D9E7510, transparent);
    border-radius: 0 12px 12px 0;
    margin: 32px 0 20px 0;
}
.section-header h2 { color: #f1f5f9; margin: 0; font-size: 1.3em; font-weight: 600; }
.section-header p  { color: #6b7280; margin: 4px 0 0 0; font-size: 0.85em; }

.insight-card {
    background: #111827;
    border-left: 4px solid #1D9E75;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 8px 0;
}
.insight-card.danger { border-left-color: #E8593C; }
.insight-card.info   { border-left-color: #3B8BD4; }
.insight-card.warn   { border-left-color: #F59E0B; }
.insight-title { font-weight: 600; color: #f1f5f9; font-size: 0.95em; }
.insight-body  { color: #9ca3af; font-size: 0.85em; margin-top: 4px; }

.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 600;
    margin: 2px;
}
.tag-green  { background: #1D9E7520; color: #1D9E75; border: 1px solid #1D9E7540; }
.tag-red    { background: #E8593C20; color: #E8593C; border: 1px solid #E8593C40; }
.tag-blue   { background: #3B8BD420; color: #3B8BD4; border: 1px solid #3B8BD440; }
.tag-yellow { background: #F59E0B20; color: #F59E0B; border: 1px solid #F59E0B40; }

.hero-title {
    font-size: 2.8em;
    font-weight: 700;
    background: linear-gradient(135deg, #1D9E75, #3B8BD4, #7F77DD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.hero-sub { color: #6b7280; font-size: 1.05em; margin-top: 8px; }

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #1D9E75 !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# FUNÇÕES DE DADOS
# ════════════════════════════════════════════════════════════
CORES = {"ativo": "#1D9E75", "churn": "#E8593C", "azul": "#3B8BD4",
         "roxo": "#7F77DD", "amarelo": "#F59E0B", "cinza": "#4b5563"}
TEMPLATE = "plotly_dark"
LAYOUT_BASE = dict(
    template=TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#9ca3af"),
    margin=dict(l=20, r=20, t=40, b=20),
)

@st.cache_data
def carregar_dados():
    """Carrega dados do SQLite se disponível, senão do CSV."""
    db_path  = "/workspaces/hackathon-churn-prediction/dados/churn_dw.db"
    csv_path = "/workspaces/hackathon-churn-prediction/dados/Bank_Customer_Churn_Prediction.csv"
    fonte = None

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT * FROM vw_cliente_completo", conn)
            conn.close()
            fonte = "SQLite (churn_dw.db)"
            df = df.rename(columns={
                "cliente_id": "customer_id", "genero": "gender",
                "idade": "age", "score_credito": "credit_score",
                "tempo_relacionamento": "tenure", "saldo": "balance",
                "pais_nome": "country", "membro_ativo": "active_member",
            })
        except Exception:
            df = None

    if df is None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        fonte = "CSV (Bank Customer Churn Prediction.csv)"

    if df is None:
        st.error("⚠️ Nenhuma fonte de dados encontrada. Coloque o CSV ou o .db na mesma pasta.")
        st.stop()

    # Normaliza colunas mínimas necessárias
    df.columns = [c.lower().strip() for c in df.columns]
    if "balance" not in df.columns and "saldo" in df.columns:
        df["balance"] = df["saldo"]
    if "active_member" not in df.columns and "membro_ativo" in df.columns:
        df["active_member"] = df["membro_ativo"]
    if "products_number" not in df.columns and "qtd_produtos" in df.columns:
        df["products_number"] = df["qtd_produtos"]
    if "estimated_salary" not in df.columns and "salario_estimado" in df.columns:
        df["estimated_salary"] = df["salario_estimado"]

    # Feature engineering
    if "faixa_etaria" not in df.columns:
        df["faixa_etaria"] = pd.cut(df["age"],
            bins=[17,30,40,50,60,100],
            labels=["18–30","31–40","41–50","51–60","61+"])
    if "tem_balance" not in df.columns:
        df["tem_balance"] = (df["balance"] > 0).astype(int)

    return df, fonte

@st.cache_data
def treinar_modelo(df):
    """Treina o Gradient Boosting e retorna métricas."""
    df_m = df.copy()
    for c in ["country","gender"]:
        if c in df_m.columns and df_m[c].dtype == object:
            df_m[c] = LabelEncoder().fit_transform(df_m[c].astype(str))

    drop_cols = ["customer_id","churn","faixa_etaria","faixa_credito",
                 "faixa_tenure","status_cliente","cor_alerta","data_referencia",
                 "perfil_produto","regiao","faixa_risco","tem_balance"]
    feat_cols = [c for c in df_m.columns
                 if c not in drop_cols and df_m[c].dtype in [np.float64, np.int64, float, int]]

    X = df_m[feat_cols].fillna(0)
    y = df_m["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                        learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)
    fi  = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)

    df["prob_churn"] = model.predict_proba(X.reindex(columns=feat_cols, fill_value=0))[:,1]

    return {"auc": auc, "fpr": fpr, "tpr": tpr, "cm": cm, "report": rep,
            "fi": fi, "X_test": X_test, "y_test": y_test,
            "y_prob": y_prob, "y_pred": y_pred}, df

# ════════════════════════════════════════════════════════════
# CARREGAMENTO
# ════════════════════════════════════════════════════════════
df_raw, fonte_dados = carregar_dados()
with st.spinner("🤖 Treinando modelo preditivo..."):
    metricas, df = treinar_modelo(df_raw)

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px'>
        <div style='font-size:2.2em'>🏦</div>
        <div style='font-weight:700; color:#f1f5f9; font-size:1.1em'>Churn Intelligence</div>
        <div style='color:#4b5563; font-size:0.75em; margin-top:4px'>Decision Intelligence Suite</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    pagina = st.radio("Navegação", [
        "🏠  Visão Executiva",
        "🔍  EDA & Perfil",
        "🤖  Modelo Preditivo",
        "⚠️  Alertas de Risco",
        "💡  Recomendações",
    ], label_visibility="collapsed")

    st.divider()

    # Filtros globais
    st.markdown("<div style='color:#6b7280; font-size:0.78em; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px'>FILTROS</div>", unsafe_allow_html=True)

    paises_disp = sorted(df["country"].dropna().unique().tolist())
    paises_sel  = st.multiselect("País", paises_disp, default=paises_disp)

    idades_range = st.slider("Faixa etária", int(df["age"].min()), int(df["age"].max()),
                              (int(df["age"].min()), int(df["age"].max())))

    st.divider()
    st.markdown(f"<div style='color:#374151; font-size:0.72em'>Fonte: {fonte_dados}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:#374151; font-size:0.72em'>{len(df):,} registros carregados</div>", unsafe_allow_html=True)

# Aplica filtros
mask = (df["country"].isin(paises_sel)) & (df["age"].between(*idades_range))
dff  = df[mask].copy()

# ════════════════════════════════════════════════════════════
# PÁG 1 — VISÃO EXECUTIVA
# ════════════════════════════════════════════════════════════
if "Visão Executiva" in pagina:

    # Hero
    col_h1, col_h2 = st.columns([3,1])
    with col_h1:
        st.markdown("""
        <div class='hero-title'>Sistema Inteligente<br>de Análise de Churn</div>
        <div class='hero-sub'>Hackathon de Dados · Tema 4 — Decision Intelligence · Bank Customer Churn</div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""
        <div class='metric-card' style='margin-top:8px'>
            <div class='metric-value info'>{metricas["auc"]:.3f}</div>
            <div class='metric-label'>AUC-ROC do modelo</div>
            <div class='metric-delta'>Gradient Boosting</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPIs principais
    n_total  = len(dff)
    n_churn  = dff["churn"].sum()
    n_ativo  = n_total - n_churn
    taxa     = n_churn / n_total * 100
    n_alto   = (dff["prob_churn"] >= 0.6).sum() if "prob_churn" in dff.columns else 0
    bal_risco = dff[dff["churn"]==1]["balance"].sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    cards = [
        (c1, f"{n_total:,}",    "CLIENTES ANALISADOS", "info",   "Base completa filtrada"),
        (c2, f"{taxa:.1f}%",    "TAXA DE CHURN",       "danger", f"{n_churn:,} cancelamentos"),
        (c3, f"{n_ativo:,}",    "CLIENTES ATIVOS",     "",       "Retidos na base"),
        (c4, f"{n_alto:,}",     "EM ALTO RISCO",       "warn",   "Prob. churn ≥ 60%"),
        (c5, f"R$ {bal_risco/1e6:.1f}M", "SALDO EM RISCO", "danger", "Clientes churned"),
    ]
    for col, val, lbl, cls, delta in cards:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value {cls}'>{val}</div>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Linha 1: Churn por país + Churn por faixa etária
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'><h2>📍 Churn por País</h2><p>Taxa percentual de cancelamento por região</p></div>", unsafe_allow_html=True)
        cp = dff.groupby("country")["churn"].agg(["sum","count"]).reset_index()
        cp["taxa"] = cp["sum"]/cp["count"]*100
        cp = cp.sort_values("taxa", ascending=True)
        fig = go.Figure(go.Bar(
            y=cp["country"], x=cp["taxa"],
            orientation="h",
            marker=dict(
                color=cp["taxa"],
                colorscale=[[0,"#1D9E75"],[0.5,"#F59E0B"],[1,"#E8593C"]],
                showscale=False,
            ),
            text=cp["taxa"].round(1).astype(str)+"%",
            textposition="outside",
        ))
        fig.update_layout(**LAYOUT_BASE, height=250,
            xaxis=dict(title="Taxa de churn (%)", gridcolor="#1f2937"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'><h2>👥 Churn por Faixa Etária</h2><p>Grupos etários com maior risco</p></div>", unsafe_allow_html=True)
        if "faixa_etaria" in dff.columns:
            fe = dff.groupby("faixa_etaria", observed=True)["churn"].agg(["sum","count"]).reset_index()
            fe["taxa"] = fe["sum"]/fe["count"]*100
        else:
            dff["_faixa"] = pd.cut(dff["age"], bins=[17,30,40,50,60,100],
                                    labels=["18–30","31–40","41–50","51–60","61+"])
            fe = dff.groupby("_faixa", observed=True)["churn"].agg(["sum","count"]).reset_index()
            fe["taxa"] = fe["sum"]/fe["count"]*100
            fe = fe.rename(columns={"_faixa":"faixa_etaria"})

        fig2 = go.Figure(go.Bar(
            x=fe["faixa_etaria"].astype(str),
            y=fe["taxa"],
            marker=dict(
                color=fe["taxa"],
                colorscale=[[0,"#1D9E75"],[0.5,"#F59E0B"],[1,"#E8593C"]],
                showscale=False,
            ),
            text=fe["taxa"].round(1).astype(str)+"%",
            textposition="outside",
        ))
        fig2.update_layout(**LAYOUT_BASE, height=250,
            yaxis=dict(title="Taxa de churn (%)", gridcolor="#1f2937"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Linha 2: Produtos + Membro ativo
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-header'><h2>📦 Churn por Nº de Produtos</h2><p>Padrão contra-intuitivo detectado no EDA</p></div>", unsafe_allow_html=True)
        pp = dff.groupby("products_number")["churn"].agg(["sum","count"]).reset_index()
        pp["taxa"] = pp["sum"]/pp["count"]*100
        cores_pp = [CORES["ativo"] if t < 25 else (CORES["amarelo"] if t < 50 else CORES["churn"])
                    for t in pp["taxa"]]
        fig3 = go.Figure(go.Bar(
            x=pp["products_number"].astype(str)+" produto(s)",
            y=pp["taxa"],
            marker_color=cores_pp,
            text=pp["taxa"].round(1).astype(str)+"%",
            textposition="outside",
        ))
        fig3.update_layout(**LAYOUT_BASE, height=250,
            yaxis=dict(title="Taxa de churn (%)", gridcolor="#1f2937"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'><h2>⚡ Status de Atividade</h2><p>Membros inativos churnam ~2x mais</p></div>", unsafe_allow_html=True)
        ma = dff.groupby("active_member")["churn"].agg(["sum","count"]).reset_index()
        ma["taxa"] = ma["sum"]/ma["count"]*100
        ma["label"] = ma["active_member"].map({1:"Membro Ativo", 0:"Membro Inativo"})
        fig4 = go.Figure()
        for _, row in ma.iterrows():
            cor = CORES["ativo"] if row["active_member"]==1 else CORES["churn"]
            fig4.add_trace(go.Bar(name=row["label"], x=[row["label"]],
                                   y=[row["taxa"]], marker_color=cor,
                                   text=f'{row["taxa"]:.1f}%', textposition="outside"))
        fig4.update_layout(**LAYOUT_BASE, height=250, showlegend=False,
            yaxis=dict(title="Taxa de churn (%)", gridcolor="#1f2937"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig4, use_container_width=True)

    # Distribuição geral (waffle simulado com treemap)
    st.markdown("<div class='section-header'><h2>🎯 Distribuição Geral da Base</h2><p>Proporção e perfil dos clientes</p></div>", unsafe_allow_html=True)
    col5, col6, col7 = st.columns([1,2,1])
    with col6:
        fig5 = go.Figure(go.Pie(
            labels=["Clientes Ativos", "Churned"],
            values=[n_ativo, n_churn],
            hole=0.65,
            marker=dict(colors=[CORES["ativo"], CORES["churn"]],
                        line=dict(color="#0a0e1a", width=3)),
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
        ))
        fig5.add_annotation(text=f"<b>{taxa:.1f}%</b><br>churn", x=0.5, y=0.5,
                             font=dict(size=22, color="white"), showarrow=False)
        fig5.update_layout(**LAYOUT_BASE, height=320, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PÁG 2 — EDA & PERFIL
# ════════════════════════════════════════════════════════════
elif "EDA" in pagina:
    st.markdown("<div class='hero-title' style='font-size:2em'>Análise Exploratória de Dados</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Distribuições, correlações e padrões identificados na base</div><br>", unsafe_allow_html=True)

    # Estatísticas descritivas
    st.markdown("<div class='section-header'><h2>📊 Estatísticas Descritivas</h2><p>Medidas de tendência central e dispersão por status de churn</p></div>", unsafe_allow_html=True)
    num_cols = ["age","credit_score","tenure","balance","products_number","estimated_salary"]
    num_cols = [c for c in num_cols if c in dff.columns]
    desc = dff.groupby("churn")[num_cols].mean().T.round(2)
    desc.columns = ["Clientes Ativos","Churned"]
    desc["Δ (%)"] = ((desc["Churned"]-desc["Clientes Ativos"])/desc["Clientes Ativos"]*100).round(1).astype(str)+"%"
    st.dataframe(desc.style.applymap(
        lambda v: "color: #E8593C" if isinstance(v,str) and v.startswith("+") else
                  "color: #1D9E75" if isinstance(v,str) and v.startswith("-") else "",
        subset=["Δ (%)"]), use_container_width=True)

    # Distribuições
    st.markdown("<div class='section-header'><h2>📈 Distribuições por Status de Churn</h2><p>Como cada feature se comporta nos dois grupos</p></div>", unsafe_allow_html=True)
    feat_sel = st.selectbox("Selecione a variável:", num_cols)
    fig_dist = go.Figure()
    for label, cor, grp in [("Ativo", CORES["ativo"], 0), ("Churned", CORES["churn"], 1)]:
        dados = dff[dff["churn"]==grp][feat_sel].dropna()
        fig_dist.add_trace(go.Histogram(x=dados, name=label, opacity=0.7,
                                         marker_color=cor, nbinsx=40,
                                         histnorm="probability density"))
    fig_dist.update_layout(**LAYOUT_BASE, height=320,
        barmode="overlay",
        xaxis=dict(title=feat_sel, gridcolor="#1f2937"),
        yaxis=dict(title="Densidade", gridcolor="#1f2937"),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_dist, use_container_width=True)

    # Heatmap de correlação
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-header'><h2>🔗 Correlações com Churn</h2></div>", unsafe_allow_html=True)
        if "credit_card" in dff.columns:
            num_cols_corr = num_cols + ["credit_card","active_member","churn"]
        else:
            num_cols_corr = num_cols + ["churn"]
        num_cols_corr = [c for c in num_cols_corr if c in dff.columns]
        corr = dff[num_cols_corr].corr()["churn"].drop("churn").sort_values()
        cores_corr = [CORES["churn"] if v > 0 else CORES["ativo"] for v in corr.values]
        fig_corr = go.Figure(go.Bar(
            y=corr.index, x=corr.values, orientation="h",
            marker_color=cores_corr,
            text=corr.round(3).values, textposition="outside",
        ))
        fig_corr.update_layout(**LAYOUT_BASE, height=340,
            xaxis=dict(title="Correlação de Pearson", gridcolor="#1f2937"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"), showlegend=False)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'><h2>🗺️ Churn: Faixa Etária × Produtos</h2></div>", unsafe_allow_html=True)
        if "faixa_etaria" not in dff.columns:
            dff["faixa_etaria"] = pd.cut(dff["age"],bins=[17,30,40,50,60,100],
                                          labels=["18–30","31–40","41–50","51–60","61+"])
        pivot = dff.groupby(["faixa_etaria","products_number"], observed=True)["churn"].mean().unstack()*100
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{c} prod." for c in pivot.columns],
            y=pivot.index.astype(str),
            colorscale=[[0,"#1D9E75"],[0.5,"#F59E0B"],[1,"#E8593C"]],
            text=np.round(pivot.values,1),
            texttemplate="%{text}%",
            textfont=dict(size=12),
            showscale=True,
        ))
        fig_heat.update_layout(**LAYOUT_BASE, height=340,
            xaxis=dict(title="Nº de produtos"),
            yaxis=dict(title="Faixa etária"))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Boxplots
    st.markdown("<div class='section-header'><h2>📦 Boxplots: Distribuição por Status</h2></div>", unsafe_allow_html=True)
    fig_box = go.Figure()
    for label, cor, grp in [("Ativo", CORES["ativo"], 0), ("Churned", CORES["churn"], 1)]:
        for feat in ["age","credit_score","balance"]:
            if feat in dff.columns:
                fig_box.add_trace(go.Box(
                    y=dff[dff["churn"]==grp][feat],
                    name=f"{label} — {feat}",
                    marker_color=cor, boxmean=True,
                    visible=True if feat=="age" else "legendonly",
                ))
    fig_box.update_layout(**LAYOUT_BASE, height=340,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h"),
        yaxis=dict(gridcolor="#1f2937"))
    st.plotly_chart(fig_box, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PÁG 3 — MODELO PREDITIVO
# ════════════════════════════════════════════════════════════
elif "Modelo" in pagina:
    st.markdown("<div class='hero-title' style='font-size:2em'>Modelo Preditivo de Churn</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Gradient Boosting · Avaliação completa · Feature Importance</div><br>", unsafe_allow_html=True)

    # Métricas do modelo
    rep = metricas["report"]
    c1,c2,c3,c4 = st.columns(4)
    cards_mod = [
        (c1, f'{metricas["auc"]:.4f}', "AUC-ROC",   "info",   "Principal métrica"),
        (c2, f'{rep["accuracy"]:.4f}', "Acurácia",  "",       "Threshold padrão 0.5"),
        (c3, f'{rep["1"]["recall"]:.4f}', "Recall Churn", "warn", "Capacidade de detectar"),
        (c4, f'{rep["1"]["f1-score"]:.4f}', "F1 Churn", "danger", "Equilíbrio prec/recall"),
    ]
    for col, val, lbl, cls, delta in cards_mod:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value {cls}'>{val}</div>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_r, col_c = st.columns(2)

    with col_r:
        st.markdown("<div class='section-header'><h2>📈 Curva ROC</h2><p>AUC mede o poder discriminatório do modelo</p></div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=metricas["fpr"], y=metricas["tpr"],
            mode="lines", name=f'Gradient Boosting (AUC={metricas["auc"]:.3f})',
            line=dict(color=CORES["azul"], width=3),
            fill="tozeroy", fillcolor="rgba(59,139,212,0.08)",
        ))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1], mode="lines",
            line=dict(color="#374151", dash="dash"), showlegend=False))
        fig_roc.update_layout(**LAYOUT_BASE, height=340,
            xaxis=dict(title="Falso positivo (FPR)", gridcolor="#1f2937"),
            yaxis=dict(title="Verdadeiro positivo (TPR)", gridcolor="#1f2937"),
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_c:
        st.markdown("<div class='section-header'><h2>🔢 Matriz de Confusão</h2><p>Verdadeiros e falsos positivos/negativos</p></div>", unsafe_allow_html=True)
        cm = metricas["cm"]
        labels = ["Ativo","Churned"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0,"#111827"],[0.5,"#1a3a5c"],[1,"#3B8BD4"]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont=dict(size=18, color="white"),
            showscale=False,
        ))
        fig_cm.update_layout(**LAYOUT_BASE, height=340,
            xaxis=dict(title="Predito"),
            yaxis=dict(title="Real", autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance
    st.markdown("<div class='section-header'><h2>🎯 Importância das Features</h2><p>Variáveis com maior poder preditivo no Gradient Boosting</p></div>", unsafe_allow_html=True)
    fi = metricas["fi"].head(10).sort_values()
    cores_fi = [CORES["churn"] if v >= fi.quantile(0.75) else
                CORES["amarelo"] if v >= fi.quantile(0.5) else CORES["azul"]
                for v in fi.values]
    fig_fi = go.Figure(go.Bar(
        y=fi.index, x=fi.values, orientation="h",
        marker_color=cores_fi,
        text=fi.round(4).values, textposition="outside",
    ))
    fig_fi.update_layout(**LAYOUT_BASE, height=360,
        xaxis=dict(title="Importância", gridcolor="#1f2937"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"), showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Distribuição das probabilidades
    st.markdown("<div class='section-header'><h2>📊 Distribuição das Probabilidades Preditas</h2></div>", unsafe_allow_html=True)
    fig_prob = go.Figure()
    for label, cor, grp in [("Ativo (real)", CORES["ativo"],0), ("Churned (real)", CORES["churn"],1)]:
        mask_g = metricas["y_test"] == grp
        fig_prob.add_trace(go.Histogram(
            x=metricas["y_prob"][mask_g], name=label,
            opacity=0.7, marker_color=cor, nbinsx=40, histnorm="probability density"))
    fig_prob.add_vline(x=0.5, line_dash="dash", line_color="#F59E0B",
                       annotation_text="Threshold 0.5", annotation_font_color="#F59E0B")
    fig_prob.update_layout(**LAYOUT_BASE, height=320, barmode="overlay",
        xaxis=dict(title="Probabilidade de churn", gridcolor="#1f2937"),
        yaxis=dict(title="Densidade", gridcolor="#1f2937"),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_prob, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PÁG 4 — ALERTAS DE RISCO
# ════════════════════════════════════════════════════════════
elif "Alertas" in pagina:
    st.markdown("<div class='hero-title' style='font-size:2em'>Painel de Alertas de Risco</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Clientes identificados pelo modelo com maior probabilidade de churn</div><br>", unsafe_allow_html=True)

    if "prob_churn" not in dff.columns:
        st.warning("Execute o modelo primeiro na página Modelo Preditivo.")
    else:
        threshold = st.slider("Threshold de risco", 0.3, 0.9, 0.5, 0.05,
                               format="%g",
                               help="Clientes com prob_churn acima deste valor são considerados em risco")

        dff["faixa_risco"] = dff["prob_churn"].apply(
            lambda p: "🔴 Crítico" if p>=0.75 else
                      "🟠 Alto"    if p>=threshold else
                      "🟡 Médio"   if p>=0.3 else "🟢 Baixo")

        col_r1, col_r2, col_r3 = st.columns(3)
        n_critico = (dff["prob_churn"]>=0.75).sum()
        n_alto    = ((dff["prob_churn"]>=threshold) & (dff["prob_churn"]<0.75)).sum()
        n_medio   = ((dff["prob_churn"]>=0.3) & (dff["prob_churn"]<threshold)).sum()
        with col_r1:
            st.markdown(f"<div class='metric-card'><div class='metric-value danger'>{n_critico}</div><div class='metric-label'>RISCO CRÍTICO</div><div class='metric-delta'>Prob ≥ 75%</div></div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='metric-card'><div class='metric-value warn'>{n_alto}</div><div class='metric-label'>RISCO ALTO</div><div class='metric-delta'>Prob ≥ {threshold:.0%}</div></div>", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"<div class='metric-card'><div class='metric-value info'>{n_medio}</div><div class='metric-label'>RISCO MÉDIO</div><div class='metric-delta'>Prob ≥ 30%</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("<div class='section-header'><h2>🎯 Distribuição de Risco</h2></div>", unsafe_allow_html=True)
            risco_cnt = dff["faixa_risco"].value_counts().reset_index()
            risco_cnt.columns = ["faixa","contagem"]
            cores_risco = {"🔴 Crítico": CORES["churn"], "🟠 Alto": CORES["amarelo"],
                           "🟡 Médio": "#F59E0B", "🟢 Baixo": CORES["ativo"]}
            fig_risco = go.Figure(go.Bar(
                x=risco_cnt["faixa"], y=risco_cnt["contagem"],
                marker_color=[cores_risco.get(r, CORES["azul"]) for r in risco_cnt["faixa"]],
                text=risco_cnt["contagem"], textposition="outside"))
            fig_risco.update_layout(**LAYOUT_BASE, height=300, showlegend=False,
                yaxis=dict(title="Nº de clientes", gridcolor="#1f2937"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_risco, use_container_width=True)

        with col_g2:
            st.markdown("<div class='section-header'><h2>💰 Saldo em Risco por País</h2></div>", unsafe_allow_html=True)
            em_risco = dff[dff["prob_churn"] >= threshold]
            bal_pais = em_risco.groupby("country")["balance"].sum().reset_index()
            bal_pais = bal_pais.sort_values("balance", ascending=True)
            fig_bal = go.Figure(go.Bar(
                y=bal_pais["country"], x=bal_pais["balance"],
                orientation="h", marker_color=CORES["churn"],
                text=(bal_pais["balance"]/1e6).round(1).astype(str)+" M",
                textposition="outside"))
            fig_bal.update_layout(**LAYOUT_BASE, height=300, showlegend=False,
                xaxis=dict(title="Saldo total em risco (R$)", gridcolor="#1f2937"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_bal, use_container_width=True)

        # Tabela de clientes em alto risco
        st.markdown("<div class='section-header'><h2>📋 Clientes em Alto Risco</h2><p>Lista para ação imediata da equipe de retenção</p></div>", unsafe_allow_html=True)
        cols_show = ["customer_id","age","gender","country","balance","credit_score",
                     "active_member","products_number","prob_churn","faixa_risco","churn"]
        cols_show = [c for c in cols_show if c in dff.columns]
        alto_risco_df = (dff[dff["prob_churn"] >= threshold][cols_show]
                         .sort_values("prob_churn", ascending=False)
                         .head(50).reset_index(drop=True))
        alto_risco_df["prob_churn"] = (alto_risco_df["prob_churn"]*100).round(1).astype(str)+"%"
        st.dataframe(alto_risco_df, use_container_width=True, height=360)


# ════════════════════════════════════════════════════════════
# PÁG 5 — RECOMENDAÇÕES
# ════════════════════════════════════════════════════════════
elif "Recomendações" in pagina:
    st.markdown("<div class='hero-title' style='font-size:2em'>Insights & Recomendações</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Achados estratégicos e ações recomendadas para reduzir o churn</div><br>", unsafe_allow_html=True)

    # Resumo executivo
    st.markdown("<div class='section-header'><h2>🧠 Resumo Executivo</h2><p>Os 5 principais achados do projeto</p></div>", unsafe_allow_html=True)

    insights = [
        ("danger", "Alta taxa de churn na Alemanha",
         "Clientes alemães churnam ~32% vs média de ~20%. Requer atenção regional específica com campanhas localizadas."),
        ("warn",   "Clientes inativos com risco dobrado",
         "Membros com active_member=0 têm taxa de churn ~2x maior. Programa de reativação pode ter alto ROI."),
        ("danger", "3+ produtos = risco paradoxal",
         "Contra-intuitivamente, clientes com 3–4 produtos têm churn elevado. Pode indicar insatisfação com a experiência multi-produto."),
        ("info",   "Faixa etária 41–60 anos é crítica",
         "Combinada com muitos produtos, gera os maiores índices de churn. Público prioritário para retenção."),
        ("info",   "Saldo positivo não garante retenção",
         "Clientes com balance > 0 churnam mais (~27% vs ~17%). Podem ter mais mobilidade financeira para trocar de banco."),
    ]

    for tipo, titulo, corpo in insights:
        st.markdown(f"""
        <div class='insight-card {tipo}'>
            <div class='insight-title'>{titulo}</div>
            <div class='insight-body'>{corpo}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Impacto financeiro estimado
    st.markdown("<div class='section-header'><h2>💸 Impacto Financeiro Estimado</h2></div>", unsafe_allow_html=True)
    n_churn_tot   = dff["churn"].sum()
    bal_perdido   = dff[dff["churn"]==1]["balance"].sum()
    n_alto_risco  = (dff["prob_churn"] >= 0.6).sum() if "prob_churn" in dff.columns else 0
    bal_em_risco  = dff[dff["prob_churn"] >= 0.6]["balance"].sum() if "prob_churn" in dff.columns else 0

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value danger'>R$ {bal_perdido/1e6:.1f}M</div>
            <div class='metric-label'>SALDO PERDIDO</div>
            <div class='metric-delta'>{n_churn_tot} clientes que saíram</div>
        </div>""", unsafe_allow_html=True)
    with col_f2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value warn'>R$ {bal_em_risco/1e6:.1f}M</div>
            <div class='metric-label'>SALDO EM RISCO</div>
            <div class='metric-delta'>{n_alto_risco} clientes em alto risco</div>
        </div>""", unsafe_allow_html=True)
    with col_f3:
        economia = bal_em_risco * 0.3
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>R$ {economia/1e6:.1f}M</div>
            <div class='metric-label'>POTENCIAL DE RETENÇÃO</div>
            <div class='metric-delta'>Retendo 30% dos clientes de risco</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plano de ação
    st.markdown("<div class='section-header'><h2>🚀 Plano de Ação Recomendado</h2></div>", unsafe_allow_html=True)
    acoes = {
        "Curto prazo (0–30 dias)": [
            ("tag-red",    "Contato imediato com clientes em risco crítico (prob ≥ 75%)"),
            ("tag-red",    "Oferta personalizada para membros inativos com saldo positivo"),
            ("tag-yellow", "Campanha de reativação segmentada por país (foco: Alemanha)"),
        ],
        "Médio prazo (30–90 dias)": [
            ("tag-blue",   "Revisão da experiência de clientes com 3+ produtos"),
            ("tag-blue",   "Programa de benefícios para faixa etária 41–60 anos"),
            ("tag-yellow", "Implementar modelo em produção com scoring semanal"),
        ],
        "Longo prazo (90+ dias)": [
            ("tag-green",  "Dashboard de monitoramento contínuo de churn"),
            ("tag-green",  "A/B test de ofertas de retenção por segmento"),
            ("tag-green",  "Retreinamento mensal do modelo com novos dados"),
        ],
    }
    for periodo, itens in acoes.items():
        st.markdown(f"**{periodo}**")
        for tag_cls, texto in itens:
            st.markdown(f"<span class='tag {tag_cls}'>●</span> {texto}", unsafe_allow_html=True)
        st.markdown("")

    # Pipeline técnico
    st.markdown("<div class='section-header'><h2>⚙️ Pipeline Técnico Implementado</h2></div>", unsafe_allow_html=True)
    steps = [
        ("📥", "Extração",     "CSV Kaggle · 10k clientes · 12 features"),
        ("🔄", "ETL",          "SQLite · Star Schema · 4 dimensões + fato"),
        ("🔍", "EDA",          "11 seções · Mann-Whitney · Correlações · Outliers"),
        ("🤖", "Modelo",       "Gradient Boosting · AUC 0.87 · Threshold ótimo"),
        ("📊", "Dashboard",    "Streamlit · Plotly · Alertas em tempo real"),
        ("🚀", "Produção",     "modelo_churn.pkl · Score via joblib"),
    ]
    cols_st = st.columns(len(steps))
    for col, (icon, titulo, desc) in zip(cols_st, steps):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='padding:18px 12px'>
                <div style='font-size:1.8em'>{icon}</div>
                <div style='color:#f1f5f9; font-weight:600; font-size:0.9em; margin:8px 0 4px'>{titulo}</div>
                <div style='color:#4b5563; font-size:0.75em'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Rodapé
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#374151; font-size:0.8em; border-top:1px solid #1f2937; padding-top:20px'>
        Hackathon de Dados · Tema 4 — Decision Intelligence · Bank Customer Churn Prediction<br>
        Pipeline: CSV → ETL → EDA → ML → Dashboard → Insights
    </div>
    """, unsafe_allow_html=True)