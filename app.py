"""
app_v2.py — Dashboard de Exportações (V2)
Reescrito e organizado como uma versão moderna do app original.

Como executar localmente:
    python -m streamlit run app_v2.py

Principais melhorias:
- Arquitetura modular (carregamento, processamento, visualização)
- Configuração por env vars / config.toml
- Tipagem e logs
- Cache apropriado com st.cache_data
- Mensagens de erro amigáveis
"""

from pathlib import Path
import logging
import os
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Configuração e logger
# ---------------------------
APP_NAME = "Vitibrasil — Exportações (V2)"
DEFAULT_DATA = Path("data/data.csv")
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
)
logger = logging.getLogger("app_v2")


# ---------------------------
# Utilitários
# ---------------------------
def humanize_number(n: float) -> str:
    """Retorna string compacta para números grandes."""
    try:
        n = float(n)
    except Exception:
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.2f}"


@st.cache_data
def read_csv_file(path: str) -> pd.DataFrame:
    """
    Lê CSV e aplica etapas básicas de normalização.
    Lança FileNotFoundError se não encontrar.
    """
    p = Path(path)
    if not p.exists():
        logger.error("CSV não encontrado em %s", path)
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(p)
    # Normalize column names (lowercase, remove espaços)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Garantir colunas básicas (se existirem)
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    for c in ["valor_exportacao", "quantidade_exportacao", "valor_exportacao_por_litro", "percentual_exportacao"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    logger.info("CSV carregado (%d linhas)", len(df))
    return df


# ---------------------------
# Processamento
# ---------------------------
def summarize_by_year(df: pd.DataFrame) -> pd.DataFrame:
    numerics = df.select_dtypes("number").columns.tolist()
    if "ano" in df.columns:
        grouped = df.groupby("ano")[numerics].sum().reset_index()
        logger.debug("Resumo por ano criado")
        return grouped
    return pd.DataFrame()


def top_countries(df: pd.DataFrame, by: str = "valor_exportacao", n: int = 10) -> pd.DataFrame:
    if "pais" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("pais", as_index=False)[["quantidade_exportacao", "valor_exportacao"]].sum()
    agg = agg.sort_values(by=by, ascending=False).head(n)
    logger.debug("Top %d países por %s calculado", n, by)
    return agg


# ---------------------------
# Visualizações
# ---------------------------
def fig_value_evolution(df_year: pd.DataFrame):
    if df_year.empty or "ano" not in df_year.columns or "valor_exportacao" not in df_year.columns:
        return None
    fig = px.line(df_year, x="ano", y="valor_exportacao", title="Evolução do Valor (US$)")
    fig.update_layout(height=520)
    return fig


def fig_top_countries_bar(df_top: pd.DataFrame):
    if df_top.empty:
        return None
    fig = px.bar(df_top, x="pais", y="valor_exportacao", title="Top Importadores (Valor)")
    fig.update_layout(height=520, xaxis_tickangle=-45)
    return fig


def fig_price_trend(df: pd.DataFrame):
    if df.empty or "ano" not in df.columns or "valor_exportacao_por_litro" not in df.columns:
        return None
    df_avg = df.groupby("ano", as_index=False)["valor_exportacao_por_litro"].mean()
    fig = px.line(df_avg, x="ano", y="valor_exportacao_por_litro", title="Preço Médio por Litro (US$)")
    fig.update_layout(height=420)
    return fig


# ---------------------------
# Interface
# ---------------------------
def sidebar_controls() -> Dict:
    st.sidebar.header("Configurações")
    data_path = st.sidebar.text_input("Caminho do CSV", value=os.getenv("DATA_PATH", str(DEFAULT_DATA)))
    sample_mode = st.sidebar.checkbox("Usar modo amostra (filtrar últimos 100 registros)", value=False)
    top_n = st.sidebar.slider("Top N países", 1, 20, 10)
    return {"data_path": data_path, "sample_mode": sample_mode, "top_n": top_n}


def render_dashboard(df: pd.DataFrame, controls: Dict):
    st.header(APP_NAME)
    st.markdown("Painel interativo com métricas e gráficos das exportações.")

    # KPIs
    total_valor = df["valor_exportacao"].sum() if "valor_exportacao" in df.columns else 0
    total_qtd = df["quantidade_exportacao"].sum() if "quantidade_exportacao" in df.columns else 0
    col1, col2 = st.columns(2)
    col1.metric("Valor total (US$)", humanize_number(total_valor))
    col2.metric("Quantidade total (L)", humanize_number(total_qtd))

    # Year summary + charts
    df_year = summarize_by_year(df)
    fig_val = fig_value_evolution(df_year)
    if fig_val:
        st.plotly_chart(fig_val, use_container_width=True)

    # Top countries
    df_top = top_countries(df, by="valor_exportacao", n=controls["top_n"])
    fig_bar = fig_top_countries_bar(df_top)
    if fig_bar:
        st.plotly_chart(fig_bar, use_container_width=True)

    # Price trend
    fig_price = fig_price_trend(df)
    if fig_price:
        st.plotly_chart(fig_price, use_container_width=True)

    # Mostrar tabela filtrável (limitada)
    with st.expander("Visualizar dados brutos (amostra)"):
        st.dataframe(df.head(500), use_container_width=True)


# ---------------------------
# Main
# ---------------------------
def main():
    st.set_page_config(page_title=APP_NAME, layout="wide")
    controls = sidebar_controls()

    try:
        df = read_csv_file(controls["data_path"])
        if controls["sample_mode"]:
            df = df.tail(100).reset_index(drop=True)
    except FileNotFoundError as e:
        st.error(f"Arquivo não encontrado: {e}")
        st.info("Coloque o CSV no repositório ou informe um caminho válido no painel lateral.")
        return
    except Exception as e:
        logger.exception("Erro ao carregar dados: %s", e)
        st.error("Erro inesperado ao processar os dados. Veja o log do servidor.")
        return

    render_dashboard(df, controls)


if __name__ == "__main__":
    main()
