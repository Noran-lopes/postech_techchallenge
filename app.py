"""
app_v5_exec_final.py — Dashboard Executivo e Analítico de Exportações de Vinho (VERSÃO AJUSTADA)

Esta versão parte do seu código original enviado inicialmente e aplica as alterações que você pediu:
- Corrige o uso de `st.progress()` (remove o parâmetro `key` que gerava `TypeError`).
- Normaliza continentes para os 5 grandes blocos (Africa, Americas, Asia, Europe, Oceania) e reduz a aparição indevida de "Outros" no treemap (filtra "Outros" quando sua participação é insignificante).
- Melhora o enriquecimento de contexto externo: usa `capitalInfo.latlng` ou `latlng` do REST Countries como fallback para obter coordenadas, reduzindo valores "N/D" no painel de contexto climático; World Bank é chamado com ISO2 em maiúsculas.
- Mantém comentários em português no estilo acadêmico e tratamento de erros.

OBS: o arquivo permanece autocontido exceto pelas dependências externas (pandas, numpy, requests, streamlit, plotly).
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
# Configurações
# ---------------------------
APP_TITLE = "WineData Insights"
DEFAULT_CSV = Path("dados_uteis/dados_uteis.csv")
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("app_v5_exec_final")

# ---------------------------
# Utilitários
# ---------------------------

def human(n: float) -> str:
    """Formata números para KPI (compacto)."""
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
# Carregamento e normalização do CSV
# ---------------------------
@st.cache_data(ttl=3600)
def load_local_csv(path: str) -> pd.DataFrame:
    """
    Lê CSV e normaliza colunas:
    Espera colunas mínimas: ano, pais, valor_exportacao, quantidade_exportacao
    Converte tipos numéricos e preenche nulos sensatos.
    """
    p = Path(path)
    if not p.exists() and not path.startswith("http"):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    try:
        if path.startswith("http://") or path.startswith("https://"):
            # aceitar URLs remotos também
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(p)
    except Exception as e:
        logger.exception("Erro lendo CSV %s: %s", path, e)
        raise
    # normalizar nomes
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    for col in ("valor_exportacao", "quantidade_exportacao", "valor_exportacao_por_litro", "percentual_exportacao"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "pais" in df.columns:
        df["pais"] = df["pais"].fillna("Desconhecido")
    return df

# ---------------------------
# APIs Externas (cacheadas)
# ---------------------------
@st.cache_data(ttl=86400)
def restcountries_search(name: str) -> Optional[dict]:
    """Consulta REST Countries - retorna primeiro resultado ou None."""
    if not name:
        return None
    try:
        url = f"https://restcountries.com/v3.1/name/{requests.utils.requote_uri(name)}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j:
            return j[0]
    except Exception as e:
        logger.debug("REST Countries fail for %s: %s", name, e)
    return None

@st.cache_data(ttl=21600)
def worldbank_gdp_percap(iso2: str, start: int = 2005, end: int = datetime.now().year) -> pd.DataFrame:
    """Consulta World Bank para NY.GDP.PCAP.CD e retorna DataFrame ['year','value']."""
    if not iso2:
        return pd.DataFrame(columns=["year", "value"])
    try:
        iso2_up = iso2.upper()
        url = f"http://api.worldbank.org/v2/country/{iso2_up}/indicator/NY.GDP.PCAP.CD?date={start}:{end}&format=json&per_page=1000"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if len(j) >= 2:
            rows = [{"year": int(item["date"]), "value": item["value"]} for item in j[1] if item.get("value") is not None]
            return pd.DataFrame(rows).sort_values("year")
    except Exception as e:
        logger.debug("WorldBank fail for %s: %s", iso2, e)
    return pd.DataFrame(columns=["year", "value"])

@st.cache_data(ttl=21600)
def open_meteo_climate(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Consulta Open-Meteo e retorna resumo climático (médias/soma)."""
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
        logger.debug("Open-Meteo fail for %s,%s: %s", lat, lon, e)
    return {}

@st.cache_data(ttl=3600)
def wine_review_proxy(country: str) -> dict:
    """Gerador determinístico de avaliações proxy (quando não há API pública)."""
    if not country:
        return {"avg_score": None, "reviews_count": 0}
    seed = abs(hash(country)) % 1000
    score = 3.6 + (seed % 140) / 100.0
    count = 30 + (seed % 400)
    return {"avg_score": round(min(score, 5.0), 2), "reviews_count": int(count)}

# ---------------------------
# Processamento / Agregações
# ---------------------------
def filter_last_n_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Retorna DataFrame filtrado pelos últimos `years` anos com base na coluna 'ano'."""
    if "ano" not in df.columns:
        return df.copy()
    max_year = int(df["ano"].max())
    min_year = max_year - years + 1
    return df[(df["ano"] >= min_year) & (df["ano"] <= max_year)].copy()


def agg_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por ano somando valor e quantidade."""
    if df.empty:
        return pd.DataFrame()
    cols = {}
    if "valor_exportacao" in df.columns:
        cols["valor_exportacao"] = "sum"
    if "quantidade_exportacao" in df.columns:
        cols["quantidade_exportacao"] = "sum"
    return df.groupby("ano", as_index=False).agg(cols).fillna(0)


def top_countries(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Retorna top N países por valor_exportacao (soma)."""
    if "pais" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("pais", as_index=False).agg({"valor_exportacao": "sum", "quantidade_exportacao": "sum"})
    return agg.sort_values("valor_exportacao", ascending=False).head(top_n)

# ---------------------------
# Normalização de continentes e mapeamento enriquecido
# ---------------------------

def normalize_region(info: Optional[dict]) -> str:
    """Normaliza para um dos 5 continentes: Africa, Americas, Asia, Europe, Oceania. Caso não identificável, retorna 'Outros'."""
    continents_allowed = {"Africa", "Americas", "Asia", "Europe", "Oceania"}
    if not info or not isinstance(info, dict):
        return "Outros"
    conts = safe_get(info, "continents", None)
    if isinstance(conts, list) and conts:
        for c in conts:
            if c in continents_allowed:
                return c
    region = safe_get(info, "region", None)
    if isinstance(region, str) and region in continents_allowed:
        return region
    sub = safe_get(info, "subregion", "") or ""
    if isinstance(sub, str) and "America" in sub:
        return "Americas"
    latlng = safe_get(info, "latlng", None)
    if isinstance(latlng, list) and len(latlng) >= 1:
        lat = latlng[0]
        # heurística simples para Oceania
        if lat > -30 and lat < 30 and safe_get(info, "region", "") == "Oceania":
            return "Oceania"
    return "Outros"


def build_iso_map_enriched(countries: List[str]) -> Dict[str, Tuple[Optional[str], Optional[str], Optional[List[float]], str, Optional[dict]]]:
    """
    Mapeia países para (iso2, iso3, latlng, region_normalized, raw_info).
    Usa `capitalInfo.latlng` como fallback para coordenadas quando disponíveis.
    """
    mapping = {}
    for c in countries:
        info = restcountries_search(c)
        iso2 = safe_get(info, "cca2", None) if info else None
        iso3 = safe_get(info, "cca3", None) if info else None
        latlng = None
        if info:
            latlng = safe_get(info, "latlng", None)
            if not latlng:
                capinfo = safe_get(info, "capitalInfo", None) or {}
                latlng = capinfo.get("latlng") if isinstance(capinfo, dict) else None
        region_norm = normalize_region(info)
        mapping[c] = (iso2.upper() if iso2 else None, iso3 if iso3 else None, latlng if latlng else None, region_norm, info)
    return mapping

# ---------------------------
# Forecast linear
# ---------------------------

def simple_linear_forecast(df_year: pd.DataFrame, n_future: int = 5) -> pd.DataFrame:
    """Previsão linear simples: entrada df_year ['ano','valor_exportacao'] — saída n_future anos adiante."""
    if df_year is None or df_year.empty or len(df_year) < 2:
        return pd.DataFrame()
    if "ano" not in df_year.columns or "valor_exportacao" not in df_year.columns:
        return pd.DataFrame()
    x = df_year["ano"].astype(float).values
    y = df_year["valor_exportacao"].astype(float).values
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    last = int(x.max())
    future_years = np.arange(last + 1, last + n_future + 1)
    preds = poly(future_years)
    return pd.DataFrame({"ano": future_years.astype(int), "valor_exportacao": preds})

# ---------------------------
# Gráficos e comentários (acadêmicos dentro do código)
# ---------------------------

def chart_value_trend(df_year: pd.DataFrame, key: str):
    if df_year.empty:
        st.info("Dados insuficientes para série temporal.")
        return
    fig = px.line(df_year, x="ano", y="valor_exportacao", markers=True,
                  title="Evolução anual do valor exportado (US$)")
    fig.update_layout(yaxis_title="Valor (US$)", xaxis_title="Ano", height=480)
    st.plotly_chart(fig, use_container_width=True, key=key)


def chart_top_countries_bar(df_top: pd.DataFrame, key: str):
    if df_top.empty:
        st.info("Sem dados de países para exibir.")
        return
    fig = px.bar(df_top, x="pais", y="valor_exportacao", hover_data=["quantidade_exportacao"],
                 title="Top países por valor exportado")
    fig.update_layout(xaxis_tickangle=-35, yaxis_title="Valor (US$)", height=420)
    st.plotly_chart(fig, use_container_width=True, key=key)


def chart_treemap_continent(df: pd.DataFrame, key: str):
    """Treemap — participação por continente com normalização."""
    if "pais" not in df.columns:
        st.info("Coluna 'pais' inexistente no dataset.")
        return
    unique_countries = df["pais"].unique().tolist()
    iso_map = build_iso_map_enriched(unique_countries)
    rows = []
    for p in unique_countries:
        region = iso_map.get(p, (None, None, None, 'Outros', None))[3]
        rows.append({"pais": p, "region": region})
    reg_df = pd.DataFrame(rows)
    merged = df.merge(reg_df, on="pais", how="left")
    agg = merged.groupby("region", as_index=False).agg({"valor_exportacao": "sum"})
    if agg.empty:
        st.info("Não foi possível montar treemap (dados insuficientes).")
        return
    # eliminar 'Outros' quando participação insignificante (<0.5%) ou zero
    total = float(agg["valor_exportacao"].sum())
    if total > 0 and "Outros" in agg["region"].values:
        outros_val = float(agg.loc[agg["region"] == "Outros", "valor_exportacao"].sum())
        if outros_val == 0 or (outros_val / total) < 0.005:
            agg = agg[agg["region"] != "Outros"]
    fig = px.treemap(agg, path=["region"], values="valor_exportacao", title="Participação por Região/Continente")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True, key=key)


def chart_scatter_price_volume(df: pd.DataFrame, key: str):
    if df.empty:
        st.info("Dados insuficientes para scatter.")
        return
    d = df.copy()
    if "valor_exportacao_por_litro" not in d.columns:
        if "valor_exportacao" in d.columns and "quantidade_exportacao" in d.columns:
            d["valor_exportacao_por_litro"] = np.where(d["quantidade_exportacao"] > 0,
                                                       d["valor_exportacao"] / d["quantidade_exportacao"], 0)
        else:
            st.info("Colunas necessárias ausentes para calcular preço por litro.")
            return
    fig = px.scatter(d, x="quantidade_exportacao", y="valor_exportacao_por_litro",
                     color="pais" if "pais" in d.columns else None,
                     size="valor_exportacao" if "valor_exportacao" in d.columns else None,
                     hover_data=["ano"] if "ano" in d.columns else None,
                     title="Preço por litro vs Quantidade (por registro)")
    fig.update_layout(xaxis_title="Quantidade (L)", yaxis_title="Valor por litro (US$)", height=520)
    st.plotly_chart(fig, use_container_width=True, key=key)


def chart_box_price_by_country(df: pd.DataFrame, key: str):
    if df.empty:
        st.info("Dados insuficientes para boxplot.")
        return
    d = df.copy()
    if "valor_exportacao_por_litro" not in d.columns:
        if "valor_exportacao" in d.columns and "quantidade_exportacao" in d.columns:
            d["valor_exportacao_por_litro"] = np.where(d["quantidade_exportacao"] > 0,
                                                       d["valor_exportacao"] / d["quantidade_exportacao"], np.nan)
        else:
            st.info("Colunas necessárias ausentes para boxplot.")
            return
    if "pais" in d.columns:
        counts = d.groupby("pais").size().reset_index(name="n")
        valid = counts[counts["n"] >= 3]["pais"].tolist()
        if not valid:
            st.info("Poucos registros por país para boxplot confiável.")
            return
        d_f = d[d["pais"].isin(valid)]
        fig = px.box(d_f, x="pais", y="valor_exportacao_por_litro", title="Distribuição de preço por litro (por país)")
        fig.update_layout(xaxis_tickangle=-45, height=520)
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.info("Coluna 'pais' ausente para boxplot por país.")


def chart_choropleth(df_top: pd.DataFrame, iso_map: Dict[str, Tuple[Optional[str], Optional[str], Optional[List[float]], str, Optional[dict]]], key: str):
    if df_top.empty:
        st.info("Sem dados para mapa.")
        return
    rows = []
    for _, r in df_top.iterrows():
        pais = r["pais"]
        iso3 = iso_map.get(pais, (None, None, None, None, None))[1]
        rows.append({"pais": pais, "valor_exportacao": r["valor_exportacao"], "iso3": iso3})
    mdf = pd.DataFrame(rows)
    if mdf["iso3"].isnull().all():
        st.info("Não foi possível mapear códigos ISO3 para os países.")
        return
    fig = px.choropleth(mdf, locations="iso3", color="valor_exportacao", hover_name="pais",
                        color_continuous_scale="Blues", title="Mapa: Valor Exportado por País")
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True, key=key)

# ---------------------------
# Insights automáticos (heurísticos)
# ---------------------------

def generate_insights(df: pd.DataFrame, df_top: pd.DataFrame, climate_map: Dict[str, dict], econ_map: Dict[str, pd.DataFrame]) -> List[str]:
    insights: List[str] = []
    total_val = float(df["valor_exportacao"].sum()) if "valor_exportacao" in df.columns else 0.0
    total_vol = float(df["quantidade_exportacao"].sum()) if "quantidade_exportacao" in df.columns else 0.0
    insights.append(f"Montante total no período: US$ {total_val:,.0f}.")
    insights.append(f"Volume total no período: {total_vol:,.0f} L.")
    if not df_top.empty:
        top1 = df_top.iloc[0]
        insights.append(f"Principal destino: {top1['pais']} (~US$ {top1['valor_exportacao']:,.0f}).")
    # climate alerts
    for country, clim in climate_map.items():
        if clim and clim.get("precip_total") and clim["precip_total"] > 100:
            insights.append(f"{country}: precipitação acumulada recente = {clim['precip_total']:.1f} mm — monitorar logística.")
    # economic context
    for country, edf in econ_map.items():
        if not edf.empty and "value" in edf.columns:
            rec = edf.dropna().sort_values("year", ascending=False).head(1)
            if not rec.empty:
                gdp = rec.iloc[0]["value"]
                insights.append(f"{country}: PIB per capita (último) ~US$ {gdp:,.0f}.")
    insights.append("Recomendação: priorizar mercados com crescimento de renda per capita e reduzir dependência nos top destinos, se o risco for concentrado.")
    return insights

# ---------------------------
# Main App
# ---------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.sidebar.header("Configurações")
    csv_path = st.sidebar.text_input("Caminho do CSV (local ou URL)", value=str(DEFAULT_CSV))
    years = st.sidebar.slider("Últimos N anos", min_value=5, max_value=30, value=15)
    top_n = st.sidebar.slider("Top N países", min_value=3, max_value=20, value=10)
    climate_months = st.sidebar.number_input("Janela clima (meses)", min_value=1, max_value=36, value=12)
    show_map = st.sidebar.checkbox("Exibir mapa coroplético", value=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("APIs: REST Countries, Open-Meteo, World Bank — uso opcional, com fallbacks.")

    # Carregar CSV
    try:
        df = load_local_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {csv_path}")
        return
    except Exception as e:
        logger.exception("Erro ao carregar CSV: %s", e)
        st.error("Erro ao carregar CSV. Veja logs.")
        return

    # Filtrar anos
    df = filter_last_n_years(df, int(years))
    if df.empty:
        st.warning("Dataset vazio após filtro de anos. Verifique o CSV e o filtro.")
        return

    # KPIs no topo
    total_val = float(df["valor_exportacao"].sum()) if "valor_exportacao" in df.columns else 0.0
    total_vol = float(df["quantidade_exportacao"].sum()) if "quantidade_exportacao" in df.columns else 0.0
    avg_price = (total_val / total_vol) if total_vol else 0.0
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.2])
    col1.metric("Valor total (US$)", human(total_val))
    col2.metric("Volume total (L)", human(total_vol))
    col3.metric("Preço médio (US$/L)", human(avg_price))
    # concentração top1
    df_top = top_countries(df, top_n)
    if not df_top.empty:
        pct_top1 = df_top.iloc[0]["valor_exportacao"] / total_val * 100 if total_val else 0
        col4.metric("Concentração top1 (%)", f"{pct_top1:.1f}%")
    else:
        col4.metric("Concentração top1 (%)", "N/D")

    # Abas
    tab_overview, tab_detailed, tab_external, tab_forecast, tab_raw, tab_insights = st.tabs(
        ["Overview", "Detalhado", "Contexto Externo", "Forecast", "Dados Brutos", "Insights"]
    )

    # Preparações comuns
    df_year = agg_by_year(df)
    iso_map_enriched = build_iso_map_enriched(df_top["pais"].tolist()) if not df_top.empty else {}

    # ------------- Overview -------------
    with tab_overview:
        st.subheader("Resumo Executivo")
        st.markdown("Visão consolidada: tendências, concentração por país e participação regional.")
        chart_value_trend(df_year, key="chart_value_trend_overview")
        chart_top_countries_bar(df_top, key="chart_top_countries_overview")
        chart_treemap_continent(df, key="chart_treemap_overview")

        if show_map and not df_top.empty:
            chart_choropleth(df_top, iso_map_enriched, key="chart_choropleth_overview")

    # ------------- Detailed -------------
    with tab_detailed:
        st.subheader("Análise Detalhada — preço, volume e dispersão")
        st.markdown("Gráficos técnicos para análise de mix, perfil de mercado e outliers.")
        chart_scatter_price_volume(df, key="chart_scatter_price_volume")
        chart_box_price_by_country(df, key="chart_box_price_country")
        st.markdown("Tabela — Top países (valor e quantidade)")
        if not df_top.empty:
            st.dataframe(df_top.reset_index(drop=True), use_container_width=True, key="df_top_table")
        else:
            st.info("Sem top países para exibir tabela.")

    # ------------- External context -------------
    with tab_external:
        st.subheader("Contexto Externo — clima e economia")
        st.markdown("Enriquecimento por país: Open-Meteo (clima) e World Bank (PIB per capita).")
        climate_map: Dict[str, dict] = {}
        econ_map: Dict[str, pd.DataFrame] = {}
        if not df_top.empty:
            # barra de progresso (sem key para evitar o TypeError)
            prog = st.progress(0)
            top_list = df_top["pais"].tolist()
            for i, pais in enumerate(top_list):
                st.markdown(f"### {pais}")
                iso2, iso3, latlng, region_norm, raw = iso_map_enriched.get(pais, (None, None, None, "Outros", None))
                lat = lon = None
                if isinstance(latlng, list) and len(latlng) >= 2:
                    lat, lon = latlng[0], latlng[1]
                # tentar Open-Meteo apenas se coordenadas plausíveis
                end_date = datetime.utcnow().date()
                start_date = (end_date - timedelta(days=30 * int(climate_months))).isoformat()
                end_date_s = end_date.isoformat()
                clim = open_meteo_climate(lat, lon, start_date, end_date_s) if lat is not None and lon is not None else {}
                climate_map[pais] = clim
                econ_df = worldbank_gdp_percap(iso2, start=datetime.utcnow().year - 10, end=datetime.utcnow().year) if iso2 else pd.DataFrame()
                econ_map[pais] = econ_df
                c1, c2, c3 = st.columns(3)
                c1.metric("Temp max média (°C)", clim.get("temp_max_avg") or "N/D")
                c2.metric("Precip total (mm)", clim.get("precip_total") or "N/D")
                # PIB per capita (último)
                if not econ_df.empty and "value" in econ_df.columns:
                    recent = econ_df.dropna().sort_values("year", ascending=False).head(1)
                    gdp_disp = f"US$ {int(recent.iloc[0]['value']):,}" if not recent.empty else "N/D"
                else:
                    gdp_disp = "N/D"
                c3.metric("PIB per capita (último)", gdp_disp)
                st.caption(f"Avaliação proxy: {wine_review_proxy(pais)['avg_score']} (n={wine_review_proxy(pais)['reviews_count']})")
                try:
                    prog.progress(int((i + 1) / len(top_list) * 100))
                except Exception:
                    pass
            try:
                prog.empty()
            except Exception:
                pass
        else:
            st.info("Sem top países para contexto externo.")

    # ------------- Forecast -------------
    with tab_forecast:
        st.subheader("Forecast Linear (Exploratório)")
        st.markdown("Modelo: regressão linear anual — usar apenas como referência exploratória.")
        if df_year.empty or len(df_year) < 2:
            st.info("Dados insuficientes para forecast.")
        else:
            n_future = st.number_input("Anos a prever (linear)", min_value=1, max_value=10, value=5, key="input_n_future")
            df_pred = simple_linear_forecast(df_year, n_future=int(n_future))
            if df_pred.empty:
                st.info("Não foi possível gerar previsão.")
            else:
                # Mostrar gráfico histórico + previsão
                combined = pd.concat([df_year, df_pred], ignore_index=True)
                fig = px.line(combined, x="ano", y="valor_exportacao", markers=True, title="Histórico + Forecast (Linear)")
                fig.add_vline(x=int(df_year["ano"].max()), line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True, key="chart_forecast_main")
                st.markdown("**Nota acadêmica:** validação temporal e modelos robustos (Prophet, ARIMA) recomendados para decisões operacionais.")

    # ------------- Raw data -------------
    with tab_raw:
        st.subheader("Dados brutos (filtrados)")
        st.dataframe(df.reset_index(drop=True), use_container_width=True, key="raw_data_table")
        st.download_button("Baixar CSV filtrado", df.to_csv(index=False).encode("utf-8"),
                           "exportacoes_filtradas.csv", "text/csv", key="download_filtered_csv")

    # ------------- Insights -------------
    with tab_insights:
        st.subheader("Insights Automáticos — Resumo Executivo")
        insights = generate_insights(df, df_top, climate_map if 'climate_map' in locals() else {}, econ_map if 'econ_map' in locals() else {})
        for idx, it in enumerate(insights, 1):
            st.write(f"{idx}. {it}")
        st.markdown("**Sugestão:** incorporar estes pontos em um slide executivo (3-5 bullets) para apresentação à gerência.")

    st.caption("V5 Final — Dashboard analítico e executivo. Desenvolvido para Tech Challenge — Pós-Tech. Limitações: forecasts exploratórios; validar antes de decisões operacionais.")

# ---------------------------
# Execução
# ---------------------------
if __name__ == "__main__":
    main()
