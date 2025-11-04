# app_v5_exec_final_improved.py — Dashboard Executivo e Analítico de Exportações de Vinho (corrigido)

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
    if not p.exists() and not path.startswith("http"):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    try:
        if path.startswith("http"):
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(p)
    except Exception as e:
        raise RuntimeError(f"Erro ao ler CSV: {e}")
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
        iso2 = iso2.upper()
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
# Processamento / Agregações
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
        mapping[c] = (iso2.upper() if iso2 else None, iso3, latlng)
    return mapping

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
# (As funções de gráficos permanecem iguais — sem mudanças funcionais)

# ---------------------------
# Main App
# ---------------------------

def normalize_region(info: Optional[dict]) -> str:
    """Normaliza o atributo de região/continente retornado pelo REST Countries para um dos 5 continentes.
    Estratégia:
    1. Usar campo 'continents' se disponível (REST Countries v3).
    2. Caso contrário, usar 'region' se estiver entre os continentes conhecidos.
    3. Se houver 'subregion' contendo 'America', mapear para 'Americas'.
    4. Caso contrário, retornar 'Outros'.
    """
    continents_allowed = {"Africa", "Americas", "Asia", "Europe", "Oceania"}
    if not info or not isinstance(info, dict):
        return "Outros"
    # Try continents field
    conts = safe_get(info, "continents", None)
    if isinstance(conts, list) and conts:
        for c in conts:
            if c in continents_allowed:
                return c
    # Try region
    region = safe_get(info, "region", None)
    if isinstance(region, str) and region in continents_allowed:
        return region
    # Try subregion heuristics
    sub = safe_get(info, "subregion", "") or ""
    if isinstance(sub, str) and "America" in sub:
        return "Americas"
    # Last resort: check latlng hemisphere (simple heuristic)
    latlng = safe_get(info, "latlng", None)
    if isinstance(latlng, list) and len(latlng) >= 1:
        lat = latlng[0]
        if lat < -60 or lat > 60:
            return "Oceania" if lat > 0 else "Americas"
    return "Outros"


def build_iso_map_enriched(countries: List[str]) -> Dict[str, Dict[str, Optional[object]]]:
    """
    Mapeia países para dict com campos: iso2, iso3, latlng, region_normalized, raw_info
    Retorna: {pais: {"iso2":..., "iso3":..., "latlng":..., "region":..., "info": ...}}
    """
    mapping: Dict[str, Dict[str, Optional[object]]] = {}
    for c in countries:
        info = restcountries_search(c)
        iso2 = safe_get(info, "cca2", None) if info else None
        iso3 = safe_get(info, "cca3", None) if info else None
        latlng = safe_get(info, "latlng", None) or safe_get(info, "capitalInfo", {}).get("latlng") if info else None
        region_norm = normalize_region(info)
        mapping[c] = {
            "iso2": iso2.upper() if iso2 else None,
            "iso3": iso3 if iso3 else None,
            "latlng": latlng if latlng else None,
            "region": region_norm,
            "info": info
        }
    return mapping


def chart_treemap_continent_improved(df: pd.DataFrame, key: str):
    """Treemap com normalização de continentes — evita categorias 'Outros' quando possível."""
    if "pais" not in df.columns:
        st.info("Coluna 'pais' inexistente no dataset.")
        return
    rows = []
    # coletar informações REST apenas uma vez por país
    unique_countries = df["pais"].unique().tolist()
    iso_enriched = build_iso_map_enriched(unique_countries)
    for p in unique_countries:
        region = iso_enriched.get(p, {}).get("region", "Outros")
        rows.append({"pais": p, "region": region})
    reg_df = pd.DataFrame(rows)
    merged = df.merge(reg_df, on="pais", how="left")
    # Agregar por região normalizada
    agg = merged.groupby("region", as_index=False).agg({"valor_exportacao": "sum"})
    if agg.empty:
        st.info("Não foi possível montar treemap (dados insuficientes).")
        return
    # remover 'Outros' se sua participação for insignificante (<0.5%) ou zero
    total = agg["valor_exportacao"].sum()
    if total > 0 and "Outros" in agg["region"].values:
        outros_val = float(agg.loc[agg["region"] == "Outros", "valor_exportacao"].sum())
        if outros_val == 0 or (outros_val / total) < 0.005:
            agg = agg[agg["region"] != "Outros"]
    fig = px.treemap(agg, path=["region"], values="valor_exportacao", title="Participação por Região/Continente")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True, key=key)


# Atualizar chart_choropleth para usar estrutura enriquecida
def chart_choropleth_improved(df_top: pd.DataFrame, iso_map_enriched: Dict[str, Dict[str, Optional[object]]], key: str):
    if df_top.empty:
        st.info("Sem dados para mapa.")
        return
    rows = []
    for _, r in df_top.iterrows():
        pais = r["pais"]
        iso3 = iso_map_enriched.get(pais, {}).get("iso3")
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
# Main App atualizado
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

    try:
        df = load_local_csv(csv_path)
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        return

    df = filter_last_n_years(df, int(years))
    if df.empty:
        st.warning("Dataset vazio após filtro de anos. Verifique o CSV e o filtro.")
        return

    total_val = float(df["valor_exportacao"].sum()) if "valor_exportacao" in df.columns else 0.0
    total_vol = float(df["quantidade_exportacao"].sum()) if "quantidade_exportacao" in df.columns else 0.0
    avg_price = (total_val / total_vol) if total_vol else 0.0
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.2])
    col1.metric("Valor total (US$)", human(total_val))
    col2.metric("Volume total (L)", human(total_vol))
    col3.metric("Preço médio (US$/L)", human(avg_price))

    df_top = top_countries(df, top_n)
    if not df_top.empty:
        pct_top1 = df_top.iloc[0]["valor_exportacao"] / total_val * 100 if total_val else 0
        col4.metric("Concentração top1 (%)", f"{pct_top1:.1f}%")
    else:
        col4.metric("Concentração top1 (%)", "N/D")

    tab_overview, tab_detailed, tab_external, tab_forecast, tab_raw, tab_insights = st.tabs(
        ["Overview", "Detalhado", "Contexto Externo", "Forecast", "Dados Brutos", "Insights"]
    )

    # Overview
    with tab_overview:
        st.subheader("Resumo Executivo")
        df_year = agg_by_year(df)
        chart_value_trend(df_year, key="chart_value_trend_overview")
        chart_top_countries_bar(df_top, key="chart_top_countries_overview")
        chart_treemap_continent_improved(df, key="chart_treemap_overview")
        if show_map and not df_top.empty:
            iso_map_enriched = build_iso_map_enriched(df_top["pais"].tolist())
            chart_choropleth_improved(df_top, iso_map_enriched, key="chart_choropleth_overview")

    # Detailed
    with tab_detailed:
        st.subheader("Análise Detalhada — preço, volume e dispersão")
        chart_scatter_price_volume(df, key="chart_scatter_price_volume")
        chart_box_price_by_country(df, key="chart_box_price_country")
        st.markdown("Tabela — Top países (valor e quantidade)")
        if not df_top.empty:
            st.dataframe(df_top.reset_index(drop=True), use_container_width=True, key="df_top_table")
        else:
            st.info("Sem top países para exibir tabela.")

    # External context (melhorias: uso de latlng capitalInfo fallback e iso_map_enriched)
    with tab_external:
        st.subheader("Contexto Externo — clima e economia")
        st.markdown("Enriquecimento por país: Open-Meteo (clima) e World Bank (PIB per capita). Foram adicionados fallbacks para lat/lon e normalização de continentes.")
        climate_map: Dict[str, dict] = {}
        econ_map: Dict[str, pd.DataFrame] = {}
        if not df_top.empty:
            iso_map_enriched = build_iso_map_enriched(df_top["pais"].tolist())
            prog = st.progress(0)
            top_list = df_top["pais"].tolist()
            for i, pais in enumerate(top_list):
                st.markdown(f"### {pais}")
                meta = iso_map_enriched.get(pais, {})
                iso2 = meta.get("iso2")
                latlng = meta.get("latlng")
                lat = lon = None
                if isinstance(latlng, list) and len(latlng) >= 2:
                    lat, lon = latlng[0], latlng[1]
                # chamar Open-Meteo apenas se tivemos coordenadas plausíveis
                clim = open_meteo_climate(lat, lon, (datetime.utcnow().date() - timedelta(days=30 * int(climate_months))).isoformat(), datetime.utcnow().date().isoformat()) if lat is not None and lon is not None else {}
                climate_map[pais] = clim or {}
                econ_df = worldbank_gdp_percap(iso2, start=datetime.utcnow().year - 10, end=datetime.utcnow().year) if iso2 else pd.DataFrame()
                econ_map[pais] = econ_df
                review = wine_review_proxy(pais)
                c1, c2, c3 = st.columns(3)
                c1.metric("Temp max média (°C)", clim.get("temp_max_avg") if clim.get("temp_max_avg") is not None else "N/D")
                c2.metric("Precip total (mm)", clim.get("precip_total") if clim.get("precip_total") is not None else "N/D")
                if not econ_df.empty and "value" in econ_df.columns:
                    recent = econ_df.dropna().sort_values("year", ascending=False).head(1)
                    gdp_disp = f"US$ {int(recent.iloc[0]['value']):,}" if not recent.empty else "N/D"
                else:
                    gdp_disp = "N/D"
                c3.metric("PIB per capita (último)", gdp_disp)
                st.caption(f"Avaliação proxy: {review['avg_score']} (n={review['reviews_count']})")
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

    # Forecast
    with tab_forecast:
        st.subheader("Forecast Linear (Exploratório)")
        if df_year.empty or len(df_year) < 2:
            st.info("Dados insuficientes para forecast.")
        else:
            n_future = st.number_input("Anos a prever (linear)", min_value=1, max_value=10, value=5, key="input_n_future")
            df_pred = simple_linear_forecast(df_year, n_future=int(n_future))
            if df_pred.empty:
                st.info("Não foi possível gerar previsão.")
            else:
                combined = pd.concat([df_year, df_pred], ignore_index=True)
                fig = px.line(combined, x="ano", y="valor_exportacao", markers=True, title="Histórico + Forecast (Linear)")
                fig.add_vline(x=int(df_year["ano"].max()), line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True, key="chart_forecast_main")
                st.markdown("**Nota acadêmica:** validar modelos mais robustos antes de usar em produção.")

    # Raw
    with tab_raw:
        st.subheader("Dados brutos (filtrados)")
        st.dataframe(df.reset_index(drop=True), use_container_width=True, key="raw_data_table")
        st.download_button("Baixar CSV filtrado", df.to_csv(index=False).encode("utf-8"), "exportacoes_filtradas.csv", "text/csv", key="download_filtered_csv")

    # Insights
    with tab_insights:
        st.subheader("Insights Automáticos — Resumo Executivo")
        insights = generate_insights(df, df_top, climate_map if 'climate_map' in locals() else {}, econ_map if 'econ_map' in locals() else {})
        for idx, it in enumerate(insights, 1):
            st.write(f"{idx}. {it}")

    st.caption("V5 — Dashboard melhorado: normalização de continentes e fallbacks no contexto externo.")


if __name__ == "__main__":
    main()
