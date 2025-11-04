"""
app_v5_exec_final.py — Dashboard Executivo de Exportações de Vinho (Versão Final)
---------------------------------------------------------------------------------
Objetivo:
 - Analisar as exportações brasileiras de vinho com base em dados do CSV e APIs externas.
 - Exibir métricas, gráficos analíticos e previsões lineares com abordagem exploratória.
 - Interface moderna, intuitiva e com comentários explicativos (padrão acadêmico).

Execução local:
    streamlit run app_v5_exec_final.py
"""

# ---------------------------
# Bibliotecas e Configuração
# ---------------------------
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

# Configuração inicial
APP_TITLE = "Vitibrasil — Dashboard Executivo (Versão Final)"
DEFAULT_CSV = Path("dados_uteis/dados_uteis.csv")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app_v5_exec_final")


# ---------------------------
# Funções Utilitárias
# ---------------------------
def human(n: float) -> str:
    """Formata números grandes para K ou M (usado nas métricas executivas)."""
    try:
        n = float(n)
    except Exception:
        return "0"
    if np.isnan(n):
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return f"{n:.2f}"


def safe_get(d: dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


# ---------------------------
# Leitura e Normalização de Dados
# ---------------------------
@st.cache_data(ttl=3600)
def load_local_csv(path: str) -> pd.DataFrame:
    """Lê e normaliza o CSV de exportações."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(p)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)

    for c in ("valor_exportacao", "quantidade_exportacao", "valor_exportacao_por_litro"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "pais" in df.columns:
        df["pais"] = df["pais"].fillna("Desconhecido")

    return df


# ---------------------------
# APIs Externas
# ---------------------------
@st.cache_data(ttl=86400)
def get_country_info(name: str) -> Optional[dict]:
    """Busca informações de país (REST Countries API)."""
    try:
        url = f"https://restcountries.com/v3.1/name/{requests.utils.requote_uri(name)}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data[0] if isinstance(data, list) and data else None
    except Exception:
        return None


@st.cache_data(ttl=21600)
def get_worldbank_gdp(iso2: str, start: int, end: int) -> pd.DataFrame:
    """Busca PIB per capita (Banco Mundial)."""
    try:
        url = f"http://api.worldbank.org/v2/country/{iso2}/indicator/NY.GDP.PCAP.CD?date={start}:{end}&format=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if len(j) >= 2:
            rows = [{"year": int(v["date"]), "value": v["value"]}
                    for v in j[1] if v.get("value")]
            return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame(columns=["year", "value"])


@st.cache_data(ttl=21600)
def get_climate(lat: float, lon: float, start: str, end: str) -> dict:
    """Obtém médias climáticas via Open-Meteo."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "start_date": start,
            "end_date": end,
            "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        d = r.json().get("daily", {})
        return {
            "temp_max_avg": np.mean(d.get("temperature_2m_max", [])) if d else None,
            "precip_total": np.sum(d.get("precipitation_sum", [])) if d else None,
        }
    except Exception:
        return {}


# ---------------------------
# Forecast (Regressão Linear Simples)
# ---------------------------
def simple_linear_forecast(df: pd.DataFrame, n_future: int = 5) -> pd.DataFrame:
    """Cria previsão linear simples de exportação (exploratória)."""
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    if "ano" not in df.columns or "valor_exportacao" not in df.columns:
        return pd.DataFrame()

    x = df["ano"].astype(float).values
    y = df["valor_exportacao"].astype(float).values

    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)

    last_year = int(x.max())
    future_years = np.arange(last_year + 1, last_year + n_future + 1)
    preds = poly(future_years)

    return pd.DataFrame({"ano": future_years.astype(int), "valor_exportacao": preds})


# ---------------------------
# Funções de Visualização
# ---------------------------
def plot_value_trend(df: pd.DataFrame):
    """Gráfico de linha: evolução anual do valor exportado."""
    if df.empty:
        st.info("Dados insuficientes.")
        return
    fig = px.line(df, x="ano", y="valor_exportacao", markers=True,
                  title="Evolução Anual do Valor Exportado (US$)")
    st.plotly_chart(fig, use_container_width=True)


def plot_top_countries(df: pd.DataFrame):
    """Gráfico de barras: principais destinos."""
    if df.empty:
        return
    fig = px.bar(df, x="pais", y="valor_exportacao",
                 title="Top Países por Valor Exportado")
    st.plotly_chart(fig, use_container_width=True)


def plot_price_volume(df: pd.DataFrame):
    """Dispersão: relação preço por litro × quantidade."""
    if "quantidade_exportacao" not in df.columns:
        return
    df["preco_medio"] = np.where(df["quantidade_exportacao"] > 0,
                                 df["valor_exportacao"] / df["quantidade_exportacao"], 0)
    fig = px.scatter(df, x="quantidade_exportacao", y="preco_medio",
                     size="valor_exportacao", color="pais",
                     title="Preço Médio (US$/L) vs Quantidade Exportada (L)")
    st.plotly_chart(fig, use_container_width=True)


def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame):
    """Linha: histórico + previsão."""
    if forecast.empty:
        st.info("Dados insuficientes para previsão.")
        return
    combined = pd.concat([df, forecast], ignore_index=True)
    fig = px.line(combined, x="ano", y="valor_exportacao",
                  markers=True, title="Histórico + Projeção Linear (5 anos)")
    fig.add_vline(x=int(df["ano"].max()), line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Função Principal
# ---------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.sidebar.header("Configurações")

    csv_path = st.sidebar.text_input("Caminho do CSV", str(DEFAULT_CSV))
    years = st.sidebar.slider("Últimos N anos", 5, 30, 15)
    top_n = st.sidebar.slider("Top N países", 3, 15, 8)
    n_future = st.sidebar.slider("Previsão (anos)", 1, 10, 5)

    # Leitura do dataset
    try:
        df = load_local_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {csv_path}")
        return
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        return

    # Filtra últimos anos
    if "ano" in df.columns:
        df = df[df["ano"] >= df["ano"].max() - years + 1]

    # KPIs principais
    total_valor = df["valor_exportacao"].sum()
    total_litros = df["quantidade_exportacao"].sum()
    preco_medio = total_valor / total_litros if total_litros else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Valor total (US$)", human(total_valor))
    c2.metric("Quantidade total (L)", human(total_litros))
    c3.metric("Preço médio (US$/L)", human(preco_medio))

    # Abas principais
    tab1, tab2, tab3 = st.tabs(["📈 Análises Gerais", "🔍 Detalhes", "🔮 Forecast"])

    with tab1:
        st.subheader("Análise Geral das Exportações")
        yearly = df.groupby("ano", as_index=False)["valor_exportacao"].sum()
        plot_value_trend(yearly)

        top = df.groupby("pais", as_index=False)["valor_exportacao"].sum()\
                .sort_values("valor_exportacao", ascending=False).head(top_n)
        plot_top_countries(top)

    with tab2:
        st.subheader("Relação Preço × Volume")
        plot_price_volume(df)

    with tab3:
        st.subheader("Previsão Linear de Exportações")
        forecast = simple_linear_forecast(yearly, n_future=n_future)
        plot_forecast(yearly, forecast)

    st.caption("Dashboard acadêmico-executivo — Pós-Tech | APIs: World Bank, REST Countries, Open-Meteo.")


# ---------------------------
# Execução
# ---------------------------
if __name__ == "__main__":
    main()
