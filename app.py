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
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(p)
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
# APIs Externas
# ---------------------------
@st.cache_data(ttl=86400)
def restcountries_search(name: str) -> Optional[dict]:
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
    if not iso2:
        return pd.DataFrame(columns=["year", "value"])
    try:
        url = f"http://api.worldbank.org/v2/country/{iso2}/indicator/NY.GDP.PCAP.CD?date={start}:{end}&format=json&per_page=1000"
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
    if not country:
        return {"avg_score": None, "reviews_count": 0}
    seed = abs(hash(country)) % 1000
    score = 3.6 + (seed % 140) / 100.0
    count = 30 + (seed % 400)
    return {"avg_score": round(min(score, 5.0), 2), "reviews_count": int(count)}

# ---------------------------
# Agregações
# ---------------------------
def filter_last_n_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if "ano" not in df.columns:
        return df.copy()
    max_year = int(df["ano"].max())
    min_year = max_year - years + 1
    return df[(df["ano"] >= min_year) & (df["ano"] <= max_year)].copy()

def agg_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = {}
    if "valor_exportacao" in df.columns:
        cols["valor_exportacao"] = "sum"
    if "quantidade_exportacao" in df.columns:
        cols["quantidade_exportacao"] = "sum"
    return df.groupby("ano", as_index=False).agg(cols).fillna(0)

def top_countries(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if "pais" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("pais", as_index=False).agg({"valor_exportacao": "sum", "quantidade_exportacao": "sum"})
    return agg.sort_values("valor_exportacao", ascending=False).head(top_n)

def build_iso_map(countries: List[str]) -> Dict[str, Tuple[Optional[str], Optional[str], Optional[List[float]]]]:
    mapping = {}
    for c in countries:
        info = restcountries_search(c)
        iso2 = safe_get(info, "cca2", None) if info else None
        iso3 = safe_get(info, "cca3", None) if info else None
        latlng = safe_get(info, "latlng", None) if info else None
        mapping[c] = (iso2.lower() if iso2 else None, iso3, latlng)
    return mapping

# ---------------------------
# Forecast linear
# ---------------------------
def simple_linear_forecast(df_year: pd.DataFrame, n_future: int = 5) -> pd.DataFrame:
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
# Gráficos
# ---------------------------

# Dicionário de tradução PT → EN para REST Countries
TRANSLATE_PT_EN = {
    "Afeganistao": "Afghanistan",
    "Africa do Sul": "South Africa",
    "Alemanha": "Germany",
    "Angola": "Angola",
    "Arabia Saudita": "Saudi Arabia",
    "Argentina": "Argentina",
    "Australia": "Australia",
    "Austria": "Austria",
    "Belgica": "Belgium",
    "Brasil": "Brazil",
    "Canada": "Canada",
    "Chile": "Chile",
    "China": "China",
    "Colombia": "Colombia",
    "Coreia do Sul": "South Korea",
    "Dinamarca": "Denmark",
    "Espanha": "Spain",
    "Estados Unidos": "United States",
    "Franca": "France",
    "Italia": "Italy",
    "Japao": "Japan",
    "Nova Zelandia": "New Zealand",
    "Portugal": "Portugal",
    "Reino Unido": "United Kingdom",
    "Suecia": "Sweden",
    "Suica": "Switzerland",
    "Uruguai": "Uruguay",
}

def chart_treemap_continent(df: pd.DataFrame, key: str):
    """Treemap — participação por continente."""
    if "pais" not in df.columns:
        st.info("Coluna 'pais' inexistente no dataset.")
        return
    rows = []
    for p in df["pais"].unique():
        nome_en = TRANSLATE_PT_EN.get(p.strip(), p)
        info = restcountries_search(nome_en)
        region = safe_get(info, "region", "Outros") if info else "Outros"
        rows.append({"pais": p, "region": region})
    reg_df = pd.DataFrame(rows)
    merged = df.merge(reg_df, on="pais", how="left")
    agg = merged.groupby("region", as_index=False).agg({"valor_exportacao": "sum"})
    if agg.empty:
        st.info("Não foi possível montar treemap (dados insuficientes).")
        return
    fig = px.treemap(agg, path=["region"], values="valor_exportacao", title="Participação por Região/Continente")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True, key=key)

# ---------------------------
# Main App
# ---------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.sidebar.header("Configurações")
    csv_path = st.sidebar.text_input("Caminho do CSV (local ou URL)", value=str(DEFAULT_CSV))
    years = st.sidebar.slider("Últimos N anos", 5, 30, 15)
    top_n = st.sidebar.slider("Top N países", 3, 20, 10)
    climate_months = st.sidebar.number_input("Janela clima (meses)", 1, 36, 12)
    show_map = st.sidebar.checkbox("Exibir mapa coroplético", value=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("APIs: REST Countries, Open-Meteo, World Bank — uso opcional, com fallbacks.")

    try:
        df = load_local_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {csv_path}")
        return
    except Exception as e:
        logger.exception("Erro ao carregar CSV: %s", e)
        st.error("Erro ao carregar CSV.")
        return

    df = filter_last_n_years(df, int(years))
    if df.empty:
        st.warning("Dataset vazio após filtro de anos.")
        return

    total_val = df["valor_exportacao"].sum()
    total_vol = df["quantidade_exportacao"].sum()
    avg_price = (total_val / total_vol) if total_vol else 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valor total (US$)", human(total_val))
    col2.metric("Volume total (L)", human(total_vol))
    col3.metric("Preço médio (US$/L)", human(avg_price))
    df_top_all = top_countries(df, top_n)
    if not df_top_all.empty:
        pct_top1 = df_top_all.iloc[0]["valor_exportacao"] / total_val * 100 if total_val else 0
        col4.metric("Concentração top1 (%)", f"{pct_top1:.1f}%")
    else:
        col4.metric("Concentração top1 (%)", "N/D")

    tab_overview, tab_detailed, tab_external = st.tabs(["Overview", "Detalhado", "Contexto Externo"])

    df_year = agg_by_year(df)
    df_top = top_countries(df, top_n)
    iso_map = build_iso_map(df_top["pais"].tolist()) if not df_top.empty else {}

    with tab_overview:
        st.subheader("Resumo Executivo")
        chart_treemap_continent(df, key="chart_treemap_overview")

    with tab_external:
        st.subheader("Contexto Externo — clima e economia")
        if not df_top.empty:
            prog = st.progress(0)  # <-- corrigido sem key
            for i, pais in enumerate(df_top["pais"]):
                st.markdown(f"### {pais}")
                iso2, iso3, latlng = iso_map.get(pais, (None, None, None))
                lat, lon = (latlng[0], latlng[1]) if latlng else (None, None)
                end_date = datetime.utcnow().date()
                start_date = (end_date - timedelta(days=30 * int(climate_months))).isoformat()
                clim = open_meteo_climate(lat, lon, start_date, end_date) if lat and lon else {}
                econ_df = worldbank_gdp_percap(iso2, start=datetime.utcnow().year - 10, end=datetime.utcnow().year) if iso2 else pd.DataFrame()
                c1, c2, c3 = st.columns(3)
                c1.metric("Temp max média (°C)", clim.get("temp_max_avg") or "N/D")
                c2.metric("Precip total (mm)", clim.get("precip_total") or "N/D")
                if not econ_df.empty and "value" in econ_df.columns:
                    recent = econ_df.dropna().sort_values("year", ascending=False).head(1)
                    gdp_disp = f"US$ {int(recent.iloc[0]['value']):,}" if not recent.empty else "N/D"
                else:
                    gdp_disp = "N/D"
                c3.metric("PIB per capita (último)", gdp_disp)
                prog.progress(int((i + 1) / len(df_top) * 100))
            prog.empty()

if __name__ == "__main__":
    main()
