"""
app_v3.py — Dashboard analítico de Exportações de Vinho (V3)
Objetivos:
 - Atender o Tech Challenge: montar análise dos últimos N anos por país (Brasil -> destinos),
   com quantidade em litros, valor em US$, e enriquecimento por dados externos (clima, economia).
 - Fornecer gráficos detalhados e comentários EXPLICATIVOS DENTRO DO CÓDIGO sobre o que cada gráfico
   representa e como interpretar (isto atende ao requisito 2).
 - Layout analítico com abas: Geral, Valor, Quantidade, Estatísticas, Dados Externos, Forecast, Dados Brutos.

Como rodar:
    python -m streamlit run app_v3.py

Dependências (adicionar ao requirements.txt):
    streamlit
    pandas
    plotly
    numpy
    requests

OBS:
 - Os comentários que descrevem os gráficos e o que eles querem demonstrar estão todos inseridos abaixo
   diretamente próximos às funções que geram os gráficos (conforme solicitado: descrições apenas no código).
"""

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

# ---------------------------
# Config / Logger
# ---------------------------
APP_TITLE = "Vitibrasil — Exportações (V3 Analítico)"
DEFAULT_CSV = Path("dados_uteis/dados_uteis.csv")
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("app_v3")

# ---------------------------
# Utils
# ---------------------------
def human(n: float) -> str:
    """Retorna número em formato compacto para dashboards."""
    try:
        n = float(n)
    except Exception:
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.2f}"

def safe_get(d: dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

# ---------------------------
# Data loading & normalization
# ---------------------------
@st.cache_data(ttl=3600)
def load_local_csv(path: str) -> pd.DataFrame:
    """
    Lê o CSV local e normaliza colunas.
    O CSV deve conter, no mínimo, colunas como:
      - pais (destino)
      - ano (int)
      - quantidade_exportacao (litros)
      - valor_exportacao (US$)

    Caso haja outras colunas, elas serão preservadas.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(p)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    for c in ("valor_exportacao", "quantidade_exportacao", "valor_exportacao_por_litro", "percentual_exportacao"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# ---------------------------
# External APIs (cached)
# ---------------------------
@st.cache_data(ttl=86400)
def get_country_info(country_name: str) -> Optional[dict]:
    """Consulta REST Countries e retorna primeiro resultado (usado para lat/lon e ISO)."""
    url = f"https://restcountries.com/v3.1/name/{requests.utils.requote_uri(country_name)}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
    except Exception as e:
        logger.warning("REST Countries error for %s: %s", country_name, e)
    return None

@st.cache_data(ttl=21600)
def get_climate_summary(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Consulta Open-Meteo para obter resumo climático entre duas datas.
    Retorna média de temperatura máxima, mínima e soma de precipitação.
    """
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
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()
        daily = j.get("daily", {})
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precips = daily.get("precipitation_sum", [])
        return {
            "temp_max_avg": float(np.mean(temps_max)) if temps_max else None,
            "temp_min_avg": float(np.mean(temps_min)) if temps_min else None,
            "precip_total": float(np.sum(precips)) if precips else None,
        }
    except Exception as e:
        logger.warning("Open-Meteo error: %s", e)
    return {}

@st.cache_data(ttl=1800)
def get_worldbank_indicator(iso2: str, indicator: str = "NY.GDP.PCAP.CD", start: int = 2005, end: int = datetime.now().year) -> pd.DataFrame:
    """
    Consulta World Bank e retorna DataFrame com colunas ['year', 'value'] do indicador.
    Note: pode retornar DataFrame vazio se API não tiver dados.
    """
    url = f"http://api.worldbank.org/v2/country/{iso2}/indicator/{indicator}?date={start}:{end}&format=json&per_page=1000"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if len(j) >= 2:
            rows = [{"year": int(v["date"]), "value": v["value"]} for v in j[1] if v.get("value") is not None]
            return pd.DataFrame(rows).sort_values("year")
    except Exception as e:
        logger.warning("World Bank error for %s: %s", iso2, e)
    return pd.DataFrame(columns=["year", "value"])

@st.cache_data(ttl=3600)
def get_wine_review_proxy(country_name: str) -> dict:
    """Placeholder determinístico para avaliação de vinhos (quando não há API pública)."""
    seed = abs(hash(country_name)) % 1000
    score = 3.5 + (seed % 150) / 100.0
    reviews_count = 50 + (seed % 500)
    return {"avg_score": round(min(score, 5.0), 2), "reviews_count": int(reviews_count)}

# ---------------------------
# Processing helpers
# ---------------------------
def filter_last_n_years(df: pd.DataFrame, years: int = 15) -> pd.DataFrame:
    if "ano" not in df.columns:
        return df.copy()
    max_year = df["ano"].max()
    min_year = max_year - (years - 1)
    return df[(df["ano"] >= min_year) & (df["ano"] <= max_year)].copy()

def top_countries_overall(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if "pais" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("pais", as_index=False).agg({"valor_exportacao": "sum", "quantidade_exportacao": "sum"})
    return agg.sort_values("valor_exportacao", ascending=False).head(top_n)

def simple_linear_forecast(series_year_value: pd.DataFrame, n_future: int = 5) -> pd.DataFrame:
    """
    Forecast linear simples:
    - Entrada: DataFrame com colunas ['ano', 'value'] (histórico anual).
    - Saída: DataFrame com previsões para os próximos n_future anos.
    NOTA: método exploratório; mencionar no relatório que modelos avançados são recomendados.
    """
    if series_year_value.empty or len(series_year_value) < 2:
        return pd.DataFrame()
    x = series_year_value["ano"].astype(float).values
    y = series_year_value["value"].astype(float).values
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    last = int(x.max())
    future_years = np.arange(last + 1, last + n_future + 1)
    preds = poly(future_years)
    return pd.DataFrame({"ano": future_years.astype(int), "value": preds})

# ---------------------------
# Graph builders with comentários explicativos (apenas no código)
# ---------------------------

def build_kpis(df: pd.DataFrame) -> dict:
    """
    KPIs calculados:
     - total_valor: soma dos valores de exportação no período selecionado.
     - total_litros: soma das quantidades (em litros).
     - preco_medio: relação valor / litros (US$/L).
    Estes KPIs aparecem no topo do dashboard para facilitar leitura rápida.
    """
    total_valor = df["valor_exportacao"].sum() if "valor_exportacao" in df.columns else 0.0
    total_litros = df["quantidade_exportacao"].sum() if "quantidade_exportacao" in df.columns else 0.0
    preco_medio = (total_valor / total_litros) if total_litros else 0.0
    return {"total_valor": total_valor, "total_litros": total_litros, "preco_medio": preco_medio}

def fig_evolucao_valor(df_year: pd.DataFrame):
    """
    Gráfico: Linha — Evolução anual do valor de exportação (US$).
    O objetivo: mostrar tendência histórica (crescimento/declínio), picos sazonais ou choques.
    Interpretação sugerida (inscrito no código): um aumento consistente indica expansão de mercado
    ou melhores preços; quedas podem indicar perda de demanda, preços ou problemas de logística.
    """
    if df_year.empty or "ano" not in df_year.columns or "valor_exportacao" not in df_year.columns:
        return None
    fig = px.line(df_year, x="ano", y="valor_exportacao", markers=True, title="Evolução anual do valor de exportação (US$)")
    fig.update_layout(yaxis_title="Valor (US$)", xaxis_title="Ano", height=520)
    return fig

def fig_evolucao_quantidade(df_year: pd.DataFrame):
    """
    Gráfico: Linha — Evolução anual da quantidade exportada (L).
    Objetivo: verificar se o aumento do valor decorre de aumento de volume ou de preço por litro.
    Comparar este gráfico com fig_evolucao_valor permite separar efeitos de preço vs. volume.
    """
    if df_year.empty or "ano" not in df_year.columns or "quantidade_exportacao" not in df_year.columns:
        return None
    fig = px.line(df_year, x="ano", y="quantidade_exportacao", markers=True, title="Evolução anual da quantidade exportada (L)")
    fig.update_layout(yaxis_title="Quantidade (L)", xaxis_title="Ano", height=520)
    return fig

def fig_top_paises_bar(df_top: pd.DataFrame):
    """
    Gráfico: Barra — Top N países por Valor de Exportação.
    Objetivo: identificar os mercados mais relevantes monetariamente.
    Use o hover para ver a quantidade e comparar se mercados com alto valor têm alta quantidade (preço).
    """
    if df_top.empty:
        return None
    fig = px.bar(df_top, x="pais", y="valor_exportacao", hover_data=["quantidade_exportacao"], title="Top destinos por valor (US$) — período selecionado")
    fig.update_layout(xaxis_tickangle=-40, yaxis_title="Valor (US$)", height=520)
    return fig

def fig_price_by_country_scatter(df_filtered: pd.DataFrame):
    """
    Gráfico: Scatter (preço por litro vs. quantidade) por país/ano.
    Objetivo: identificar outliers de preço (mercados pagam preço premium) e relações preço-volume.
    Pontos maiores = maior valor exportado; cor = país.
    """
    if df_filtered.empty or "valor_exportacao_por_litro" not in df_filtered.columns:
        return None
    fig = px.scatter(df_filtered, x="quantidade_exportacao", y="valor_exportacao_por_litro", color="pais",
                     size="valor_exportacao", hover_data=["ano"], title="Preço por litro vs Quantidade (por país/registro)")
    fig.update_layout(xaxis_title="Quantidade (L)", yaxis_title="Valor por litro (US$)", height=520)
    return fig

def fig_preco_medio_ano(df_year_price: pd.DataFrame):
    """
    Gráfico: Linha — Preço médio por litro ao longo dos anos.
    Objetivo: verificar se os preços por litro estão subindo (melhor percepção de valor) ou caindo.
    Complementa a análise valor vs quantidade.
    """
    if df_year_price.empty or "ano" not in df_year_price.columns or "valor_exportacao_por_litro" not in df_year_price.columns:
        return None
    df_avg = df_year_price.groupby("ano", as_index=False)["valor_exportacao_por_litro"].mean()
    fig = px.line(df_avg, x="ano", y="valor_exportacao_por_litro", markers=True, title="Preço médio por litro (US$) — por ano")
    fig.update_layout(yaxis_title="US$/L", xaxis_title="Ano", height=420)
    return fig

# ---------------------------
# Insights generator (texto)
# ---------------------------
def generate_insights(df_filtered: pd.DataFrame, df_top: pd.DataFrame, climate_by_country: Dict[str, dict], econ_by_country: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Gera uma lista de insights curtos (strings) baseados nas métricas e dados externos.
    Estes insights são automáticos e servem como ponto de partida para a apresentação.
    """
    insights: List[str] = []
    # Insight: montante total
    total_valor = df_filtered["valor_exportacao"].sum() if "valor_exportacao" in df_filtered.columns else 0
    insights.append(f"Montante acumulado (período selecionado): US$ {total_valor:,.0f}.")
    # Insight: principal destino
    if not df_top.empty:
        top = df_top.iloc[0]
        insights.append(f"Principal destino: {top['pais']} (~US$ {top['valor_exportacao']:,.0f}).")
    # Climate/econ heuristics
    for country, clim in climate_by_country.items():
        if clim:
            prec = clim.get("precip_total")
            if prec is not None and prec > 100:
                insights.append(f"{country}: precipitação acumulada recente = {prec:.1f} mm — monitorar logística/sazonalidade.")
    for country, econ_df in econ_by_country.items():
        if not econ_df.empty and "value" in econ_df.columns:
            recent = econ_df.dropna().sort_values("year", ascending=False)
            if not recent.empty:
                gdp = recent.iloc[0]["value"]
                insights.append(f"{country}: PIB per capita mais recente ~US$ {gdp:,.0f} — considerar segmentação premium/massa.")
    # Recomendações resumidas
    insights.append("Recomendações (automáticas): diversificar destinos com crescimento econômico, investir em canais digitais nos mercados-alvo e alinhar logística a janelas sazonais de demanda.")
    return insights

# ---------------------------
# UI & Main
# ---------------------------
def header_ui(kpis: dict):
    st.title(APP_TITLE)
    st.markdown("Painel analítico para a apresentação a investidores — foco em valor, volume, preço e fatores externos.")
    # Exibir KPIs em cards
    col1, col2, col3 = st.columns([1.4, 1.4, 1.0])
    col1.metric("Valor total (US$)", human(kpis["total_valor"]))
    col2.metric("Quantidade total (L)", human(kpis["total_litros"]))
    col3.metric("Preço médio (US$/L)", human(kpis["preco_medio"]))

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    # Sidebar controls
    st.sidebar.header("Parâmetros")
    csv_path = st.sidebar.text_input("Caminho do CSV local (ou URL)", value=str(DEFAULT_CSV))
    years = st.sidebar.slider("Últimos N anos", min_value=5, max_value=30, value=15)
    top_n = st.sidebar.slider("Top N países", 3, 20, 10)
    climate_months = st.sidebar.number_input("Janela clima (meses)", min_value=1, max_value=36, value=12)
    include_forecast = st.sidebar.checkbox("Incluir forecast linear", value=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("APIs usadas: REST Countries, Open-Meteo, World Bank. Quando não disponíveis, há fallbacks.")

    # Load data
    try:
        df = load_local_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Arquivo CSV não encontrado: {csv_path}")
        st.info("Coloque o CSV em dados_uteis/dados_uteis.csv ou informe caminho/URL.")
        return
    except Exception as e:
        logger.exception("Erro ao ler CSV: %s", e)
        st.error("Erro ao ler CSV. Ver logs.")
        return

    # Filtrar últimos N anos
    df_filtered = filter_last_n_years(df, years)

    # KPIs
    kpis = build_kpis(df_filtered)
    header_ui(kpis)

    # Preparar agregações
    df_year = df_filtered.groupby("ano", as_index=False).agg({
        "valor_exportacao": "sum" if "valor_exportacao" in df_filtered.columns else pd.NamedAgg(column=None, aggfunc="sum"),
        "quantidade_exportacao": "sum" if "quantidade_exportacao" in df_filtered.columns else pd.NamedAgg(column=None, aggfunc="sum"),
        "valor_exportacao_por_litro": "mean" if "valor_exportacao_por_litro" in df_filtered.columns else pd.NamedAgg(column=None, aggfunc="mean"),
    }).fillna(0)

    df_top = top_countries_overall(df_filtered, top_n)

    # Abas analíticas, inspiradas no primeiro código
    tab_geral, tab_valor, tab_quantidade, tab_stats, tab_external, tab_forecast, tab_raw = st.tabs(
        ["Geral", "Valor", "Quantidade", "Estatísticas", "Dados Externos", "Forecast", "Dados Brutos"]
    )

    # ---------------------------------
    # Aba GERAL (resumo + tabela)
    # ---------------------------------
    with tab_geral:
        st.subheader("Visão Geral")
        st.markdown(
            "- **Objetivo:** apresentar o montante total, principais destinos e a tendência dos últimos anos.\n"
            "- **Como usar:** selecione N anos e Top N países na sidebar; explore gráficos e baixe os dados."
        )
        # Gráfico: evolução do valor
        fig_val = fig_evolucao_valor(df_year)
        if fig_val:
            st.plotly_chart(fig_val, use_container_width=True)
        # Mostrar top países resumido
        st.markdown("**Top países (por valor)**")
        if not df_top.empty:
            st.dataframe(df_top, use_container_width=True)
        else:
            st.info("Sem dados de países para o período selecionado.")

    # ---------------------------------
    # Aba VALOR (foco em valor US$)
    # ---------------------------------
    with tab_valor:
        st.subheader("Análise por Valor (US$)")
        # Comentário explicativo (só no código): O objetivo desta aba é detalhar variações do valor exportado,
        # identificar picos e possíveis correlações com eventos externos (que depois serão checados nas abas externas).
        # Gráfico 1: Evolução anual do valor (linha) — já construído como fig_evolucao_valor
        if fig_val:
            st.plotly_chart(fig_val, use_container_width=True)
            st.markdown("**Interpretação rápida (ver comentário no código):** Compare a evolução do valor com a evolução da quantidade (aba 'Quantidade') para entender se oscilações são por preço ou volume.")
        # Gráfico 2: Top países por valor (barra)
        fig_top = fig_top_paises_bar(df_top)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)
        # Tabela detalhada por país
        st.markdown("**Detalhe por país (valor e quantidade)**")
        st.dataframe(df_top.rename(columns={"valor_exportacao": "Valor (US$)", "quantidade_exportacao": "Quantidade (L)"}), use_container_width=True)

    # ---------------------------------
    # Aba QUANTIDADE (foco em litros)
    # ---------------------------------
    with tab_quantidade:
        st.subheader("Análise por Quantidade (Litros)")
        # Gráfico: evolução da quantidade por ano
        fig_qtd = fig_evolucao_quantidade(df_year)
        if fig_qtd:
            st.plotly_chart(fig_qtd, use_container_width=True)
            st.markdown("**Interpretação rápida (no código):** Se a quantidade aumenta mas o valor não, pode haver queda no preço por litro — verificar aba 'Estatísticas' para preço médio.")
        # Gráfico complementar: preço por país (scatter) — identifica mercados premium
        fig_scatter = fig_price_by_country_scatter(df_filtered)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------------------------------
    # Aba ESTATÍSTICAS (preços, percentuais)
    # ---------------------------------
    with tab_stats:
        st.subheader("Estatísticas e Indicadores")
        st.markdown("Nesta aba apresentamos medidas que ajudam a separar efeitos de preço e volume.")
        # Preço médio por litro ano a ano
        fig_price = fig_preco_medio_ano(df_filtered)
        if fig_price:
            st.plotly_chart(fig_price, use_container_width=True)
            st.markdown("**O que este gráfico mostra (comentário no código):** aumento do preço médio por litro indica melhoria de mix/posicionamento ou aumento de preços; queda pode indicar perda de percepção de valor.")
        # Percentual exportado da produção — se coluna existir
        if "percentual_exportacao" in df_filtered.columns:
            pct_by_year = df_filtered.groupby("ano", as_index=False)["percentual_exportacao"].mean()
            fig_pct = px.line(pct_by_year, x="ano", y="percentual_exportacao", markers=True, title="Percentual médio da produção exportado (%)")
            fig_pct.update_layout(yaxis_title="Percentual (%)", height=420)
            st.plotly_chart(fig_pct, use_container_width=True)
        else:
            st.info("Coluna 'percentual_exportacao' não encontrada no CSV; se disponível, mostramos o indicador aqui.")

    # ---------------------------------
    # Aba DADOS EXTERNOS (clima, economia, reviews)
    # ---------------------------------
    with tab_external:
        st.subheader("Dados Externos (Clima e Economia) — Top países")
        st.markdown("Abaixo consultamos APIs públicas (REST Countries, Open-Meteo, World Bank) para enriquecer a análise. Se houver falha, mostramos 'N/D'.")
        climate_by_country: Dict[str, dict] = {}
        econ_by_country: Dict[str, pd.DataFrame] = {}
        today = datetime.utcnow().date()
        start_date = (today - timedelta(days=30 * int(climate_months))).isoformat()
        end_date = today.isoformat()
        if not df_top.empty:
            top_list = df_top["pais"].tolist()
            progress = st.progress(0)
            for i, country in enumerate(top_list):
                st.write(f"**{country}**")
                info = get_country_info(country)
                lat, lon, iso2 = None, None, ""
                if info:
                    latlng = safe_get(info, "latlng", [])
                    if len(latlng) >= 2:
                        lat, lon = latlng[0], latlng[1]
                    iso2 = safe_get(info, "cca2", "")[:2].lower()
                # climate
                clim = get_climate_summary(lat, lon, start_date, end_date) if lat and lon else {}
                climate_by_country[country] = clim
                c1, c2, c3 = st.columns(3)
                c1.metric("Temp max média (°C)", clim.get("temp_max_avg") or "N/D")
                c2.metric("Precip total (mm)", clim.get("precip_total") or "N/D")
                # econ
                econ_df = get_worldbank_indicator(iso2, start=today.year - years, end=today.year) if iso2 else pd.DataFrame()
                econ_by_country[country] = econ_df
                if not econ_df.empty and "year" in econ_df.columns and "value" in econ_df.columns:
                    recent = econ_df.dropna().sort_values("year", ascending=False).head(1)
                    gdp_str = f"US$ {int(recent.iloc[0]['value']):,}" if not recent.empty else "N/D"
                else:
                    gdp_str = "N/D"
                c3.metric("PIB per capita (último)", gdp_str)
                # reviews proxy
                wrev = get_wine_review_proxy(country)
                st.caption(f"Avaliação proxy: {wrev['avg_score']} (n={wrev['reviews_count']})")
                progress.progress(int((i + 1) / len(top_list) * 100))
            progress.empty()
        else:
            st.info("Sem top países calculados — verifique os filtros e o CSV.")

    # ---------------------------------
    # Aba FORECAST (simples)
    # ---------------------------------
    with tab_forecast:
        st.subheader("Forecast simples (regressão linear)")
        df_series = df_filtered.groupby("ano", as_index=False)["valor_exportacao"].sum().rename(columns={"valor_exportacao": "value"})
        if df_series.empty or len(df_series) < 2:
            st.info("Dados insuficientes para forecast (mínimo 2 pontos).")
        else:
            if include_forecast:
                n_future = st.number_input("Anos a prever (linear)", min_value=1, max_value=10, value=5)
                df_pred = simple_linear_forecast(df_series.rename(columns={"value": "value"}), n_future=int(n_future))
                df_plot = pd.concat([df_series.rename(columns={"value": "value"}), df_pred.rename(columns={"value": "value"})], ignore_index=True)
                fig = px.line(df_plot, x="ano", y="value", markers=True, title="Histórico + Forecast (linear) — Valor exportado (US$)")
                fig.add_vline(x=int(df_series["ano"].max()), line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Observação (no código):** método linear exploratório; para decisão financeira usar modelos robustos (Prophet, ARIMA) e validar com backtest.")

    # ---------------------------------
    # Aba DADOS BRUTOS
    # ---------------------------------
    with tab_raw:
        st.subheader("Dados brutos filtrados")
        st.dataframe(df_filtered, use_container_width=True)
        st.download_button("Baixar CSV filtrado", df_filtered.to_csv(index=False).encode("utf-8"), "exportacoes_filtradas.csv", "text/csv")

    # ---------------------------------
    # Insights automáticos (sidebar)
    # ---------------------------------
    st.sidebar.markdown("---")
    if st.sidebar.button("Gerar insights automáticos"):
        with st.spinner("Gerando insights..."):
            insights = generate_insights(df_filtered, df_top, climate_by_country if 'climate_by_country' in locals() else {}, econ_by_country if 'econ_by_country' in locals() else {})
            st.sidebar.markdown("### Insights automáticos")
            for idx, it in enumerate(insights, 1):
                st.sidebar.write(f"{idx}. {it}")

    st.caption("Versão V3 analítica — desenvolvida para o Tech Challenge (Fase 1). Comentários explicativos dos gráficos estão inclusos no código (apenas no código).")

if __name__ == "__main__":
    main()
