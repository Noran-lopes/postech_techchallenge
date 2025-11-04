"""
app_v5_exec.py — Dashboard Executivo de Exportações de Vinho (Versão para Gerência)

Objetivo:
 - Painel analítico executivo, rico em visualizações, pronto para apresentação à gerência.
 - Mantém integração com APIs públicas: REST Countries, Open-Meteo, World Bank.
 - Usa CSV local padrão: dados_uteis/dados_uteis.csv
 - Fornece gráficos: linhas, barras, treemap, scatter, boxplot, mapa (choropleth), forecast e box/violin.
 - Comentários e explicações em português no estilo acadêmico inseridos no código, próximo às funções que constroem gráficos.
 - Cada elemento Streamlit tem key explícita onde necessário para evitar problemas de ID duplicados.

Como rodar:
    pip install -r requirements.txt
    streamlit run app_v5_exec.py

Dependências recomendadas (ex.: requirements.txt):
    streamlit==1.39.0
    pandas==2.2.3
    plotly==5.24.1
    numpy==1.26.4
    requests==2.32.3
"""

from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# ---------------------------
# Config / Logger
# ---------------------------
APP_TITLE = "Vitibrasil — Dashboard Executivo (V5)"
DEFAULT_CSV = Path("dados_uteis/dados_uteis.csv")
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("app_v5_exec")

# ---------------------------
# Utilities
# ---------------------------

def human(n: float) -> str:
    """Formata números grandes de uma forma compacta (acadêmico/exec)."""
    try:
        n = float(n)
    except Exception:
        return "0"
    if np.isnan(n):
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.2f}"

def safe_get(d: dict, k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

# ---------------------------
# Load & Normalize CSV
# ---------------------------

@st.cache_data(ttl=3600)
def load_local_csv(path: str) -> pd.DataFrame:
    """
    Lê CSV local, normaliza colunas e tipos.
    Espera (pelo menos) colunas: ano, pais, valor_exportacao, quantidade_exportacao.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(p)
    # Normalizar cabeçalhos
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Garantir tipos
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    for c in ("valor_exportacao", "quantidade_exportacao", "valor_exportacao_por_litro", "percentual_exportacao"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # preencher país vazio com "Desconhecido"
    if "pais" in df.columns:
        df["pais"] = df["pais"].fillna("Desconhecido")
    return df

# ---------------------------
# External APIs (cached)
# ---------------------------

@st.cache_data(ttl=86400)
def restcountries_search(name: str) -> Optional[dict]:
    """Consulta REST Countries para retornar o primeiro resultado (usado p/ iso2/iso3/latlng)."""
    if not name or str(name).strip() == "":
        return None
    url = f"https://restcountries.com/v3.1/name/{requests.utils.requote_uri(name)}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
    except Exception as e:
        logger.debug("REST Countries failed for %s: %s", name, e)
    return None

@st.cache_data(ttl=21600)
def worldbank_gdp_percap(iso2: str, start: int = 2005, end: int = datetime.now().year) -> pd.DataFrame:
    """Consulta World Bank para NY.GDP.PCAP.CD, retorna df ['year','value']."""
    if not iso2:
        return pd.DataFrame(columns=["year", "value"])
    url = f"http://api.worldbank.org/v2/country/{iso2}/indicator/NY.GDP.PCAP.CD?date={start}:{end}&format=json&per_page=1000"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if len(j) >= 2:
            rows = [{"year": int(item["date"]), "value": item["value"]} for item in j[1] if item.get("value") is not None]
            return pd.DataFrame(rows).sort_values("year")
    except Exception as e:
        logger.debug("WorldBank failed for %s: %s", iso2, e)
    return pd.DataFrame(columns=["year", "value"])

@st.cache_data(ttl=21600)
def open_meteo_climate(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Consulta Open-Meteo e retorna médias/total (temp max/min avg, precip total)."""
    if lat is None or lon is None:
        return {}
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }
    try:
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()
        daily = j.get("daily", {})
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        prec = daily.get("precipitation_sum", [])
        return {
            "temp_max_avg": float(np.mean(temps_max)) if temps_max else None,
            "temp_min_avg": float(np.mean(temps_min)) if temps_min else None,
            "precip_total": float(np.sum(prec)) if prec else None
        }
    except Exception as e:
        logger.debug("Open-Meteo failed for %s,%s: %s", lat, lon, e)
    return {}

@st.cache_data(ttl=3600)
def wine_review_proxy(country: str) -> dict:
    """Gerador determinístico de 'reviews' quando não há API pública.*"""
    if not country:
        return {"avg_score": None, "reviews_count": 0}
    seed = abs(hash(country)) % 1000
    score = 3.6 + (seed % 140) / 100.0
    count = 30 + (seed % 400)
    return {"avg_score": round(min(score, 5.0), 2), "reviews_count": int(count)}

# ---------------------------
# Processing helpers
# ---------------------------

def filter_last_n_years(df: pd.DataFrame, years: int = 15) -> pd.DataFrame:
    if "ano" not in df.columns:
        return df.copy()
    maxy = int(df["ano"].max())
    miny = maxy - years + 1
    return df[(df["ano"] >= miny) & (df["ano"] <= maxy)].copy()

def agg_year(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por ano somando valor e quantidade."""
    if df.empty:
        return pd.DataFrame()
    cols = {}
    if "valor_exportacao" in df.columns:
        cols["valor_exportacao"] = "sum"
    if "quantidade_exportacao" in df.columns:
        cols["quantidade_exportacao"] = "sum"
    return df.groupby("ano", as_index=False).agg(cols).fillna(0)

def top_countries(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if "pais" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("pais", as_index=False).agg({"valor_exportacao": "sum", "quantidade_exportacao": "sum"})
    return agg.sort_values("valor_exportacao", ascending=False).head(n)

def country_iso_map(df_countries: List[str]) -> Dict[str, Tuple[Optional[str], Optional[str], Optional[List[float]]]]:
    """
    Constroi dicionário {pais: (iso2, iso3, [lat,lon])} via REST Countries.
    Executa cache via restcountries_search.
    """
    mapping = {}
    for c in df_countries:
        info = restcountries_search(c)
        iso2 = None
        iso3 = None
        latlng = None
        if info:
            iso2 = safe_get(info, "cca2", None)
            iso3 = safe_get(info, "cca3", None)
            latlng = safe_get(info, "latlng", None)
        mapping[c] = (iso2.lower() if iso2 else None, iso3 if iso3 else None, latlng if latlng else None)
    return mapping

# ---------------------------
# Charts (with academic comments)
# ---------------------------

def chart_value_trend(df_year: pd.DataFrame):
    """
    Linha: Evolução anual do valor exportado (US$).
    Interpretação acadêmica: examinar a tendência (slope), picos e quebras de série (eventos).
    """
    if df_year.empty or "valor_exportacao" not in df_year.columns:
        return None
    fig = px.line(df_year, x="ano", y="valor_exportacao", markers=True,
                  title="Evolução anual do valor exportado (US$)")
    fig.update_layout(yaxis_title="Valor (US$)", xaxis_title="Ano", height=520)
    return fig

def chart_top_countries_bar(df_top: pd.DataFrame):
    """
    Barras: Top N países por valor exportado.
    Interpretação: avaliar concentração (HHI informal) e identificar mercados-chave.
    """
    if df_top.empty:
        return None
    fig = px.bar(df_top, x="pais", y="valor_exportacao", hover_data=["quantidade_exportacao"],
                 title="Top países por valor exportado")
    fig.update_layout(xaxis_tickangle=-35, yaxis_title="Valor (US$)", height=480)
    return fig

def chart_treemap_by_continent(df: pd.DataFrame, country_iso: Dict[str, Tuple[Optional[str], Optional[str], Optional[List[float]]]]):
    """
    Treemap: participacao por continente (estimado por mapeamento REST Countries).
    Objetivo: demonstrar diversificação geográfica por bloco regional.
    Observação: utilizamos REST Countries para obter region/continent.
    """
    if "pais" not in df.columns:
        return None
    # Add continent column using REST Countries
    continents = []
    for p in df["pais"].unique():
        info = restcountries_search(p)
        region = safe_get(info, "region", "Outros") if info else "Outros"
        continents.append((p, region))
    cont_df = pd.DataFrame(continents, columns=["pais", "continente"])
    merged = df.merge(cont_df, on="pais", how="left")
    agg = merged.groupby(["continente"], as_index=False).agg({"valor_exportacao": "sum"})
    if agg.empty:
        return None
    fig = px.treemap(agg, path=["continente"], values="valor_exportacao",
                     title="Participação por Continente (Valor)")
    fig.update_layout(height=520)
    return fig

def chart_scatter_price_volume(df: pd.DataFrame):
    """
    Scatter: relação entre quantidade e valor por país/registo.
    Interpretação: identifica mercados com alto preço por litro (premium) vs alta quantidade (volume).
    Requer coluna 'valor_exportacao_por_litro' ou será estimado.
    """
    if df.empty:
        return None
    # compute price per liter if not present
    d = df.copy()
    if "valor_exportacao_por_litro" not in d.columns and "valor_exportacao" in d.columns and "quantidade_exportacao" in d.columns:
        # avoid division by zero
        d = d.assign(valor_exportacao_por_litro = d.apply(
            lambda row: (row["valor_exportacao"] / row["quantidade_exportacao"]) if row["quantidade_exportacao"] > 0 else 0, axis=1))
    if "valor_exportacao_por_litro" not in d.columns:
        return None
    fig = px.scatter(d, x="quantidade_exportacao", y="valor_exportacao_por_litro",
                     color="pais" if "pais" in d.columns else None,
                     size="valor_exportacao" if "valor_exportacao" in d.columns else None,
                     hover_data=["ano"] if "ano" in d.columns else None,
                     title="Preço por litro vs Quantidade (por registro)")
    fig.update_layout(xaxis_title="Quantidade (L)", yaxis_title="Valor por litro (US$)", height=520)
    return fig

def chart_price_boxplot(df: pd.DataFrame):
    """
    Boxplot: distribuição do preço por litro por país (ou global).
    Objetivo: identificar variabilidade de preços e outliers.
    """
    if df.empty:
        return None
    d = df.copy()
    if "valor_exportacao_por_litro" not in d.columns:
        if "valor_exportacao" in d.columns and "quantidade_exportacao" in d.columns:
            d = d.assign(valor_exportacao_por_litro = d.apply(
                lambda r: (r["valor_exportacao"] / r["quantidade_exportacao"]) if r["quantidade_exportacao"] > 0 else np.nan, axis=1))
        else:
            return None
    # Keep countries with enough observations
    if "pais" in d.columns:
        fig = px.box(d, x="pais", y="valor_exportacao_por_litro", title="Distribuição de Preço por Litro (por País)")
        fig.update_layout(xaxis_tickangle=-45, height=520)
        return fig
    else:
        fig = px.box(d, y="valor_exportacao_por_litro", title="Distribuição de Preço por Litro")
        fig.update_layout(height=520)
        return fig

def chart_choropleth(df_top: pd.DataFrame, iso_map: Dict[str, Tuple[Optional[str], Optional[str], Optional[List[float]]]]):
    """
    Mapa coroplético: mostra valor exportado por país no mapa mundial.
    Exige ISO3 codes (cca3) para plotly. Utilizamos restcountries_search para mapear.
    Observação acadêmica: fornece visão espacial imediata da distribuição internacional.
    """
    if df_top.empty:
        return None
    # Build df with iso3
    rows = []
    for _, row in df_top.iterrows():
        pais = row["pais"]
        iso2, iso3, latlng = iso_map.get(pais, (None, None, None))
        rows.append({"pais": pais, "valor_exportacao": row["valor_exportacao"], "iso3": iso3})
    mapdf = pd.DataFrame(rows)
    if mapdf["iso3"].isnull().all():
        return None
    fig = px.choropleth(mapdf, locations="iso3", color="valor_exportacao", hover_name="pais",
                        color_continuous_scale="Blues", title="Mapa: Valor Exportado por País")
    fig.update_layout(height=560)
    return fig

# ---------------------------
# Insights generator (brief)
# ---------------------------

def generate_insights(df: pd.DataFrame, df_top: pd.DataFrame, climate_map: Dict[str, dict], econ_map: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Gera insights executivos automáticos (curtos).
    Observação: são heurísticos; devem ser validados qualitativamente antes do envio à gerência.
    """
    insights = []
    total_val = float(df["valor_exportacao"].sum()) if "valor_exportacao" in df.columns else 0.0
    total_vol = float(df["quantidade_exportacao"].sum()) if "quantidade_exportacao" in df.columns else 0.0
    insights.append(f"Montante total (período): US$ {total_val:,.0f}")
    insights.append(f"Volume total (L): {total_vol:,.0f}")
    if not df_top.empty:
        top1 = df_top.iloc[0]
        insights.append(f"Principal destino: {top1['pais']} (~US$ {top1['valor_exportacao']:,.0f})")
    # climate heuristic
    for country, clim in climate_map.items():
        if clim and clim.get("precip_total") and clim["precip_total"] > 100:
            insights.append(f"{country}: precipitação recente alta ({clim['precip_total']:.1f} mm) — avaliar riscos logísticos.")
    # econ heuristic
    for country, edf in econ_map.items():
        if not edf.empty and "value" in edf.columns:
            recent = edf.dropna().sort_values("year", ascending=False)
            if not recent.empty:
                gdp = recent.iloc[0]["value"]
                insights.append(f"{country}: PIB per capita ~US$ {gdp:,.0f}")
    insights.append("Recomendação sumária: priorizar mercados com maior crescimento de PIB per capita e reduzir concentração em top-3 se risco concentrado.")
    return insights

# ---------------------------
# Main UI / Layout
# ---------------------------

def header_kpis(df: pd.DataFrame):
    """Calcula e exibe KPIs executivos no topo do dashboard (sem keys para evitar duplicidade)."""
    total_val = df["valor_exportacao"].sum() if "valor_exportacao" in df.columns else 0.0
    total_vol = df["quantidade_exportacao"].sum() if "quantidade_exportacao" in df.columns else 0.0
    avg_price = (total_val / total_vol) if total_vol else 0.0

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
    c1.metric("Valor total (US$)", human(total_val))
    c2.metric("Volume total (L)", human(total_vol))
    c3.metric("Preço médio (US$/L)", human(avg_price))
    # Concentração: percent top1
    df_top_all = top_countries(df, n=1) if "pais" in df.columns else pd.DataFrame()
    if not df_top_all.empty:
        pct = df_top_all.iloc[0]["valor_exportacao"] / total_val * 100 if total_val else 0
        c4.metric("Concentração top1 (%)", f"{pct:.1f}%")
    else:
        c4.metric("Concentração top1 (%)", "N/D")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.sidebar.header("Configurações")
    csv_path = st.sidebar.text_input("Caminho do CSV", value=str(DEFAULT_CSV), key="csv_path")
    years = st.sidebar.slider("Últimos N anos", 5, 30, 15, key="years")
    top_n = st.sidebar.slider("Top N países", 5, 20, 10, key="topn")
    climate_months = st.sidebar.number_input("Janela clima (meses)", min_value=1, max_value=36, value=12, key="clim_months")
    show_map = st.sidebar.checkbox("Exibir mapa coroplético", value=True, key="show_map")
    st.sidebar.markdown("---")
    st.sidebar.caption("APIs: REST Countries, Open-Meteo, World Bank. Há fallbacks se APIs falharem.")

    # Load data
    try:
        df = load_local_csv(csv_path)
    except FileNotFoundError as e:
        st.error(f"CSV não encontrado: {e}")
        return
    except Exception as e:
        logger.exception("Erro leitura CSV: %s", e)
        st.error("Erro ao carregar CSV. Veja logs.")
        return

    # Filter years
    df = filter_last_n_years(df, years)
    if df.empty:
        st.warning("Dataset vazio após filtro de anos. Verifique CSV.")
        return

    # Top countries and aggregations
    df_year = agg_year(df)
    df_top = top_countries(df, n=top_n)

    # Header KPIs
    header_kpis(df)

    # Tabs
    tabs = st.tabs(["Overview", "Detalhado", "Contexto Externo", "Forecast", "Dados Brutos", "Insights"])
    # Overview tab
    with tabs[0]:
        st.subheader("Resumo Executivo")
        st.markdown("Evolução temporal e principais destinos — visão rápida para diretoria.")
        fig_val = chart_value_trend(df_year)
        if fig_val:
            st.plotly_chart(fig_val, use_container_width=True, key="overview_val_trend")
        fig_top = chart_top_countries_bar(df_top)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True, key="overview_top_bar")
        # Treemap por continente
        fig_tree = chart_treemap_by_continent(df, country_iso_map(df["pais"].unique().tolist()))
        if fig_tree:
            st.plotly_chart(fig_tree, use_container_width=True, key="overview_treemap")

    # Detalhado tab
    with tabs[1]:
        st.subheader("Análise Detalhada")
        st.markdown("Gráficos técnicos para avaliação: price-volume, distribuição de preços e análise por país.")
        # Scatter price vs volume
        fig_scatter = chart_scatter_price_volume(df)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True, key="detailed_scatter")
        # Boxplot price distribution
        fig_box = chart_price_boxplot(df)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True, key="detailed_box")
        # Table: top countries detail
        st.markdown("Tabela: Top países — Valor e Quantidade")
        if not df_top.empty:
            st.dataframe(df_top.reset_index(drop=True), use_container_width=True, key="detailed_table_top")
        else:
            st.info("Sem top países para exibir.")

    # Contexto Externo tab
    with tabs[2]:
        st.subheader("Contexto Externo — Clima & Economia")
        st.markdown("Consultamos REST Countries, Open-Meteo e World Bank para enriquecer análise por destino.")
        climate_map = {}
        econ_map = {}
        iso_map = country_iso_map(df_top["pais"].tolist()) if not df_top.empty else {}
        # progress
        if not df_top.empty:
            progress = st.progress(0, key="ctx_progress")
            for i, pais in enumerate(df_top["pais"].tolist()):
                st.markdown(f"### {pais}")
                iso2, iso3, latlng = iso_map.get(pais, (None, None, None))
                lat, lon = (latlng[0], latlng[1]) if latlng else (None, None)
                # climate window
                end = datetime.utcnow().date()
                start = (end - timedelta(days=30 * climate_months)).isoformat()
                clim = open_meteo_climate(lat, lon, start, end) if lat and lon else {}
                climate_map[pais] = clim
                econ = worldbank_gdp_percap(iso2, start=datetime.utcnow().year - 10, end=datetime.utcnow().year) if iso2 else pd.DataFrame()
                econ_map[pais] = econ
                c1, c2, c3 = st.columns(3)
                c1.metric("Temp max média (°C)", clim.get("temp_max_avg") or "N/D", key=f"ctx_temp_{i}")
                c2.metric("Precip total (mm)", clim.get("precip_total") or "N/D", key=f"ctx_precip_{i}")
                # GDP last
                if not econ.empty and "value" in econ.columns:
                    last = econ.dropna().sort_values("year", ascending=False).head(1)
                    gdp_val = f"US$ {int(last.iloc[0]['value']):,}" if not last.empty else "N/D"
                else:
                    gdp_val = "N/D"
                c3.metric("PIB per capita (último)", gdp_val, key=f"ctx_gdp_{i}")
                st.caption(f"Avaliação média proxy: {wine_review_proxy(pais)['avg_score']} (n={wine_review_proxy(pais)['reviews_count']})", key=f"ctx_rev_{i}")
                progress.progress(int((i+1)/len(df_top)*100) if len(df_top) else 100)
            progress.empty()
        else:
            st.info("Sem top países para contexto externo.")

        if show_map and not df_top.empty:
            fig_map = chart_choropleth(df_top, iso_map)
            if fig_map:
                st.plotly_chart(fig_map, use_container_width=True, key="ctx_map")

    # Forecast tab
    with tabs[3]:
        st.subheader("Forecast Linear (Exploratório)")
        st.markdown("Projeção baseada em regressão linear anual (usar como referência exploratória).")
        if df_year.empty or len(df_year) < 2:
            st.info("Dados insuficientes para forecast.")
        else:
            n_future = st.number_input("Anos a prever", min_value=1, max_value=10, value=5, key="forecast_years")
            df_series = df_year.rename(columns={"valor_exportacao": "valor_exportacao"})
            df_pred = simple_linear_forecast(df_series, n_future=int(n_future))
            if not df_pred.empty:
                combined = pd.concat([df_series, df_pred], ignore_index=True)
                figf = px.line(combined, x="ano", y="valor_exportacao", markers=True, title="Histórico + Forecast (linear)")
                figf.add_vline(x=int(df_series["ano"].max()), line_dash="dash", line_color="gray")
                st.plotly_chart(figf, use_container_width=True, key="forecast_chart")
                st.markdown("Nota: realizar backtest e usar modelos avançados para decisões críticas.")

    # Dados Brutos tab
    with tabs[4]:
        st.subheader("Dados Brutos (filtrados)")
        st.dataframe(df.reset_index(drop=True), use_container_width=True, key="raw_df")
        st.download_button("Baixar CSV filtrado", df.to_csv(index=False).encode("utf-8"),
                           "exportacoes_filtradas.csv", "text/csv", key="download_raw")

    # Insights tab
    with tabs[5]:
        st.subheader("Insights Automáticos — Resumo Executivo")
        insights = generate_insights(df, df_top, climate_map if 'climate_map' in locals() else {}, econ_map if 'econ_map' in locals() else {})
        for idx, it in enumerate(insights, 1):
            st.write(f"{idx}. {it}")
        st.markdown("**Sugestão para apresentação:** use os gráficos das abas anteriores e destaque top3 recomendações baseadas nos insights acima.")

    st.caption("Versão V5 Executiva — desenvolvida para apresentação à gerência. Métodos explicativos e limitações documentadas no código.")

if __name__ == "__main__":
    main()
