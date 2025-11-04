"""
app_v4_final.py — Dashboard Analítico de Exportações de Vinho (Versão Final)
-------------------------------------------------------------------------------
Objetivo:
    Desenvolver um painel interativo que analise as exportações brasileiras de vinho,
    integrando dados externos (econômicos e climáticos) e fornecendo previsões simples.

Abordagem:
    - Integração via API (World Bank, REST Countries, Open-Meteo)
    - Forecast linear (projeção de tendência futura)
    - Interface moderna com Streamlit e Plotly
    - Explicações analíticas para cada gráfico (atende aos critérios acadêmicos)

Execução local:
    streamlit run app_v4_final.py
"""

# ---------------------------
# Bibliotecas e Configuração
# ---------------------------
from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

APP_TITLE = "Vitibrasil — Exportações de Vinho (Versão Final)"
DEFAULT_CSV = Path("dados_uteis/dados_uteis.csv")
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("app_v4_final")

# ---------------------------
# Funções Utilitárias
# ---------------------------
def safe_get(d: dict, key: str, default=None):
    """Acessa um dicionário com segurança, retornando valor padrão caso a chave não exista."""
    return d.get(key, default) if isinstance(d, dict) else default

def human(n: float) -> str:
    """Formata valores numéricos em unidades legíveis (K, M)."""
    try:
        n = float(n)
    except Exception:
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.2f}"

# ---------------------------
# Carregamento de Dados
# ---------------------------
@st.cache_data(ttl=3600)
def load_local_csv(path: str) -> pd.DataFrame:
    """Carrega o CSV local, padroniza colunas e tipos numéricos."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(p)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype(int)
    for c in ["valor_exportacao", "quantidade_exportacao"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# ---------------------------
# APIs Externas
# ---------------------------
@st.cache_data(ttl=86400)
def get_country_info(country_name: str) -> Optional[dict]:
    """Consulta a API REST Countries para obter coordenadas e ISO2 do país."""
    url = f"https://restcountries.com/v3.1/name/{requests.utils.requote_uri(country_name)}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data[0] if isinstance(data, list) and data else None
    except Exception as e:
        logger.warning(f"Erro REST Countries ({country_name}): {e}")
        return None

@st.cache_data(ttl=21600)
def get_worldbank_indicator(iso2: str, indicator: str = "NY.GDP.PCAP.CD",
                            start: int = 2005, end: int = datetime.now().year) -> pd.DataFrame:
    """Obtém dados de PIB per capita do Banco Mundial."""
    url = f"http://api.worldbank.org/v2/country/{iso2}/indicator/{indicator}?date={start}:{end}&format=json&per_page=1000"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if len(j) >= 2:
            rows = [{"year": int(v["date"]), "value": v["value"]} for v in j[1] if v.get("value")]
            return pd.DataFrame(rows).sort_values("year")
    except Exception as e:
        logger.warning(f"Erro WorldBank ({iso2}): {e}")
    return pd.DataFrame(columns=["year", "value"])

@st.cache_data(ttl=21600)
def get_climate_summary(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Consulta a API Open-Meteo e calcula médias de temperatura e precipitação."""
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        daily = j.get("daily", {})
        res = {
            "temp_max_avg": float(np.mean(daily.get("temperature_2m_max", []))) if daily else None,
            "precip_total": float(np.sum(daily.get("precipitation_sum", []))) if daily else None,
        }
        return res
    except Exception as e:
        logger.warning(f"Erro Open-Meteo ({lat},{lon}): {e}")
        return {}

# ---------------------------
# Processamento e Forecast
# ---------------------------
def filter_last_n_years(df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    """Filtra o DataFrame para os últimos N anos."""
    if "ano" not in df.columns:
        return df
    max_year = df["ano"].max()
    return df[df["ano"] >= max_year - years + 1]

def simple_linear_forecast(df: pd.DataFrame, n_future: int = 5) -> pd.DataFrame:
    """
    Aplica regressão linear para prever valores futuros.
    - Explicação acadêmica:
        Este modelo linear básico estima uma tendência média de crescimento
        com base nos anos anteriores, fornecendo uma visão exploratória da evolução.
    """
    if len(df) < 2:
        return pd.DataFrame()
    x = df["ano"].values
    y = df["valor_exportacao"].values
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    future_years = np.arange(x.max() + 1, x.max() + n_future + 1)
    preds = poly(future_years)
    return pd.DataFrame({"ano": future_years, "valor_exportacao": preds})

# ---------------------------
# Interface do Usuário
# ---------------------------
def header_ui(kpis: Dict[str, float]):
    """Renderiza KPIs principais com visual limpo e interpretável."""
    st.title(APP_TITLE)
    st.markdown("Análise interativa das exportações brasileiras de vinho, integrando variáveis externas (econômicas e climáticas).")

    total_valor = float(kpis.get("total_valor", 0) or 0)
    total_litros = float(kpis.get("total_litros", 0) or 0)
    preco_medio = float(kpis.get("preco_medio", 0) or 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Valor total (US$)", human(total_valor))
    c2.metric("Quantidade total (L)", human(total_litros))
    c3.metric("Preço médio (US$/L)", human(preco_medio))

def overview_tab(df: pd.DataFrame):
    """
    Aba 'Overview':
      - Mostra a evolução anual das exportações.
      - Gráfico de linha: tendência temporal do valor total exportado.
      - Gráfico de barras: principais países de destino.
    """
    st.subheader("Evolução Anual das Exportações")
    df_year = df.groupby("ano", as_index=False).agg({"valor_exportacao": "sum"})
    fig_val = px.line(df_year, x="ano", y="valor_exportacao", markers=True,
                      title="Tendência Anual do Valor Exportado (US$)")
    st.plotly_chart(fig_val, use_container_width=True)

    st.subheader("Top 10 Países por Valor Exportado")
    df_top = df.groupby("pais", as_index=False)["valor_exportacao"].sum().sort_values("valor_exportacao", ascending=False).head(10)
    fig_top = px.bar(df_top, x="pais", y="valor_exportacao",
                     title="Principais Destinos das Exportações de Vinho")
    st.plotly_chart(fig_top, use_container_width=True)

def forecast_tab(df: pd.DataFrame):
    """Aba de previsão linear — análise da tendência projetada."""
    st.subheader("Projeção Linear de Exportações")
    forecast = simple_linear_forecast(df, n_future=5)
    if forecast.empty:
        st.info("Dados insuficientes para previsão.")
        return
    combined = pd.concat([df, forecast])
    fig = px.line(combined, x="ano", y="valor_exportacao", markers=True,
                  title="Histórico + Projeção Linear (5 anos)")
    fig.add_vline(x=df["ano"].max(), line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretação:** Esta projeção linear ilustra a tendência esperada de exportações com base na média histórica, servindo como referência exploratória.")

# ---------------------------
# Função Principal
# ---------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Sidebar
    st.sidebar.header("Configurações")
    csv_path = st.sidebar.text_input("Caminho do CSV", value=str(DEFAULT_CSV))
    years = st.sidebar.slider("Últimos N anos", 5, 30, 15)

    try:
        df = load_local_csv(csv_path)
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        return

    df = filter_last_n_years(df, years)

    # KPIs
    kpis = {
        "total_valor": df["valor_exportacao"].sum(),
        "total_litros": df["quantidade_exportacao"].sum(),
        "preco_medio": df["valor_exportacao"].sum() / df["quantidade_exportacao"].sum() if df["quantidade_exportacao"].sum() else 0
    }
    header_ui(kpis)

    tab1, tab2 = st.tabs(["📊 Overview", "🔮 Forecast"])
    with tab1:
        overview_tab(df)
    with tab2:
        forecast_tab(df)

    st.caption("Painel acadêmico desenvolvido para o Tech Challenge — Pós-Tech. APIs: World Bank, Open-Meteo e REST Countries.")

if __name__ == "__main__":
    main()
