"""
app_v4.py — Dashboard analítico de Exportações de Vinho (V4)

Resumo:
 - Mantém o layout moderno e analítico do código anterior (V2/V3).
 - Corrige problema StreamlitDuplicateElementId adicionando 'key' exclusivos a elementos.
 - Inclui comentários explicativos em Português (padrão entrega acadêmica) junto das funções de plot.
 - Usa CSV local por padrão: dados_uteis/dados_uteis.csv
 - Integra APIs públicas: REST Countries, Open-Meteo, World Bank (com fallbacks).
 - Forecast linear simples (padrão 5 anos), com opção de alterar anos no sidebar.

Como executar:
    python -m streamlit run app_v4.py

Dependências (requirements.txt):
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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# ---------------------------
# Configurações globais
# ---------------------------
APP_TITLE = "Vitibrasil — Exportações (V4 Analítico)"
DEFAULT_CSV = Path("dados_uteis/dados_uteis.csv")
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("app_v4")

# ---------------------------
# Utilitários (genéricos)
# ---------------------------

def human(n: float) -> str:
    """Formatador numérico compacto para exibição em KPIs (ex.: 1.2M, 45.6K)."""
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
    """Acesso seguro a dicionários, retorna default se não existir."""
    return d.get(key, default) if isinstance(d, dict) else default

# ---------------------------
# Leitura e normalização do CSV
# ---------------------------

@st.cache_data(ttl=3600)
def load_local_csv(path: str) -> pd.DataFrame:
    """
    Lê e normaliza o CSV de entrada.
    Espera colunas (nomes case-insensitive):
      - ano
      - pais
      - quantidade_exportacao (L)
      - valor_exportacao (US$)
      - valor_exportacao_por_litro (US$/L) [opcional]
      - percentual_exportacao [%] [opcional]

    A função:
      - normaliza nomes das colunas para snake_case lowercase;
      - converte tipos para numerics onde aplicável;
      - preenche NAs com 0 onde faz sentido.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(p)
    # Normalizar nomes de coluna
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Tipos
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    for c in ("valor_exportacao", "quantidade_exportacao", "valor_exportacao_por_litro", "percentual_exportacao"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# ---------------------------
# APIs externas (cacheadas)
# ---------------------------

@st.cache_data(ttl=86400)
def get_country_info(country_name: str) -> Optional[dict]:
    """
    REST Countries API (v3) — busca informações do país (latlng, cca2, capital, etc.)
    Retorna o primeiro resultado válido ou None se falhar.
    """
    if not country_name:
        return None
    url = f"https://restcountries.com/v3.1/name/{requests.utils.requote_uri(country_name)}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
    except Exception as e:
        logger.warning("REST Countries error for %s: %s", country_name, e)
    return None

@st.cache_data(ttl=21600)
def get_climate_summary(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Consulta Open-Meteo para obter estatísticas diárias entre start_date e end_date.
    Retorna dicionário com médias e soma de precipitação.
    """
    if lat is None or lon is None:
        return {}
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
        logger.warning("Open-Meteo error for %s,%s: %s", lat, lon, e)
    return {}

@st.cache_data(ttl=1800)
def get_worldbank_indicator(iso2: str, indicator: str = "NY.GDP.PCAP.CD",
                            start: int = 2005, end: int = datetime.now().year) -> pd.DataFrame:
    """
    Busca indicador do World Bank (formato JSON).
    Retorna DataFrame com colunas ['year', 'value'] ou DataFrame vazio.
    """
    if not iso2:
        return pd.DataFrame(columns=["year", "value"])
    url = f"http://api.worldbank.org/v2/country/{iso2}/indicator/{indicator}?date={start}:{end}&format=json&per_page=1000"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if len(j) >= 2:
            rows = [{"year": int(v["date"]), "value": v["value"]} for v in j[1] if v.get("value") is not None]
            return pd.DataFrame(rows).sort_values("year")
    except Exception as e:
        logger.warning("WorldBank error for %s: %s", iso2, e)
    return pd.DataFrame(columns=["year", "value"])

@st.cache_data(ttl=3600)
def get_wine_review_proxy(country_name: str) -> dict:
    """
    Placeholder determinístico para avaliações de vinho por país.
    Utilizado quando não existe API pública livre para avaliações.
    """
    if not country_name:
        return {"avg_score": None, "reviews_count": 0}
    seed = abs(hash(country_name)) % 1000
    score = 3.5 + (seed % 150) / 100.0
    reviews_count = 50 + (seed % 500)
    return {"avg_score": round(min(score, 5.0), 2), "reviews_count": int(reviews_count)}

# ---------------------------
# Processamento local
# ---------------------------

def filter_last_n_years(df: pd.DataFrame, years: int = 15) -> pd.DataFrame:
    """Filtra o dataframe para conter apenas os últimos `years` anos com base na coluna 'ano'."""
    if "ano" not in df.columns:
        return df.copy()
    max_year = int(df["ano"].max())
    min_year = max_year - (years - 1)
    return df[(df["ano"] >= min_year) & (df["ano"] <= max_year)].copy()

def top_countries_overall(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Retorna os top_n países ordenados por soma de valor_exportacao (decrescente)."""
    if "pais" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("pais", as_index=False).agg({
        "valor_exportacao": "sum",
        "quantidade_exportacao": "sum"
    })
    return agg.sort_values("valor_exportacao", ascending=False).head(top_n)

def build_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula KPIs principais: total valor, total litros, preço médio."""
    total_valor = float(df["valor_exportacao"].sum()) if "valor_exportacao" in df.columns else 0.0
    total_litros = float(df["quantidade_exportacao"].sum()) if "quantidade_exportacao" in df.columns else 0.0
    preco_medio = (total_valor / total_litros) if total_litros else 0.0
    return {"total_valor": total_valor, "total_litros": total_litros, "preco_medio": preco_medio}

# ---------------------------
# Forecast (linear)
# ---------------------------

def simple_linear_forecast(series_year_value: pd.DataFrame, n_future: int = 5) -> pd.DataFrame:
    """
    Previsão linear simples (regressão polinomial de grau 1).
    Entrada: DataFrame com colunas ['ano', 'value'].
    Saída: DataFrame com previsões para os próximos n_future anos.
    Nota acadêmica: método exploratório; recomenda-se usar modelos robustos para decisões financeiras.
    """
    if series_year_value.empty or len(series_year_value) < 2:
        return pd.DataFrame()
    x = series_year_value["ano"].astype(float).values
    y = series_year_value["value"].astype(float).values
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    last_year = int(x.max())
    future_years = np.arange(last_year + 1, last_year + n_future + 1)
    preds = poly(future_years)
    return pd.DataFrame({"ano": future_years.astype(int), "value": preds})

# ---------------------------
# Construção de gráficos e comentários analíticos (padrão acadêmico)
# ---------------------------
# Cada função abaixo contém comentários explicativos sobre o objetivo do gráfico,
# sua interpretação e o que deve ser observado pelo avaliador/investidor.
# Comentários são apenas dentro do código, conforme solicitado.

def fig_evolucao_valor(df_year: pd.DataFrame):
    """
    Gráfico: Linha — Evolução anual do valor de exportação (US$).
    Objetivo analítico:
      - Mostrar tendência de longo prazo do montante em US$.
      - Identificar anos atípicos (picos/quedas) que merecem investigação qualitativa.
      - Permitir comparação temporal com outras métricas (volume, preço médio).
    Como interpretar:
      - Aumento persistente: possível expansão de mercado ou melhoria do preço por litro.
      - Queda: investigar perda de mercado, variações cambiais ou problemas logísticos.
    """
    if df_year.empty or "ano" not in df_year.columns or "valor_exportacao" not in df_year.columns:
        return None
    fig = px.line(df_year, x="ano", y="valor_exportacao", markers=True,
                  title="Evolução anual do valor de exportação (US$)")
    fig.update_layout(yaxis_title="Valor (US$)", xaxis_title="Ano", height=520)
    return fig

def fig_evolucao_quantidade(df_year: pd.DataFrame):
    """
    Gráfico: Linha — Evolução anual da quantidade exportada (litros).
    Objetivo analítico:
      - Verificar tendência de volume exportado.
      - Separar efeitos de preço (valor) dos efeitos de quantidade (volume).
    Interpretação combinada:
      - Se valor cresce e quantidade também cresce: expansão real de mercado.
      - Se valor cresce e quantidade cai: aumento de preço por litro (melhor mix).
    """
    if df_year.empty or "ano" not in df_year.columns or "quantidade_exportacao" not in df_year.columns:
        return None
    fig = px.line(df_year, x="ano", y="quantidade_exportacao", markers=True,
                  title="Evolução anual da quantidade exportada (L)")
    fig.update_layout(yaxis_title="Quantidade (L)", xaxis_title="Ano", height=520)
    return fig

def fig_top_paises_bar(df_top: pd.DataFrame):
    """
    Gráfico: Barras — Top N países por valor de exportação.
    Objetivo analítico:
      - Identificar concentração de receita em poucos destinos.
      - Apoiar recomendações de diversificação geográfica.
    Uso prático:
      - Hover mostra quantidade associada para verificar se o alto valor é por volume ou preço.
    """
    if df_top.empty:
        return None
    fig = px.bar(df_top, x="pais", y="valor_exportacao", hover_data=["quantidade_exportacao"],
                 title="Top destinos por valor (US$) — período selecionado")
    fig.update_layout(xaxis_tickangle=-35, yaxis_title="Valor (US$)", height=520)
    return fig

def fig_preco_medio_ano(df_filtered: pd.DataFrame):
    """
    Gráfico: Linha — Preço médio por litro ao longo dos anos.
    Objetivo analítico:
      - Avaliar evolução do preço médio praticado (valor/volume).
      - Diagnosticar mudanças de mix de produto ou posicionamento de preço.
    Observação:
      - Exige coluna 'valor_exportacao_por_litro' ou cálculo implícito (valor/quantidade).
    """
    if df_filtered.empty:
        return None
    # Tentar usar coluna explícita; caso contrário, calcular preço médio por ano
    if "valor_exportacao_por_litro" in df_filtered.columns and df_filtered["valor_exportacao_por_litro"].sum() > 0:
        df_price = df_filtered.groupby("ano", as_index=False)["valor_exportacao_por_litro"].mean()
        ycol = "valor_exportacao_por_litro"
        ytitle = "US$/L (média)"
    elif "valor_exportacao" in df_filtered.columns and "quantidade_exportacao" in df_filtered.columns:
        tmp = df_filtered.groupby("ano", as_index=False).agg({
            "valor_exportacao": "sum",
            "quantidade_exportacao": "sum"
        }).query("quantidade_exportacao > 0")
        if tmp.empty:
            return None
        tmp["valor_por_litro"] = tmp["valor_exportacao"] / tmp["quantidade_exportacao"]
        df_price = tmp[["ano", "valor_por_litro"]].rename(columns={"valor_por_litro": "valor_exportacao_por_litro"})
        ycol = "valor_exportacao_por_litro"
        ytitle = "US$/L (calculado)"
    else:
        return None
    fig = px.line(df_price, x="ano", y=ycol, markers=True, title="Preço médio por litro — por ano")
    fig.update_layout(yaxis_title=ytitle, xaxis_title="Ano", height=420)
    return fig

def fig_price_vs_volume_scatter(df_filtered: pd.DataFrame):
    """
    Gráfico: Scatter — Relação entre quantidade exportada e preço por litro por registro.
    Objetivo analítico:
      - Identificar mercados com preço premium (alto US$/L) e seu volume.
      - Detectar outliers (mercados muito caros ou muito baratos).
    Observação:
      - Requer 'valor_exportacao_por_litro' e 'quantidade_exportacao'; caso contrário, retorna None.
    """
    if df_filtered.empty:
        return None
    if "valor_exportacao_por_litro" not in df_filtered.columns or "quantidade_exportacao" not in df_filtered.columns:
        return None
    fig = px.scatter(df_filtered, x="quantidade_exportacao", y="valor_exportacao_por_litro",
                     color="pais" if "pais" in df_filtered.columns else None,
                     size="valor_exportacao" if "valor_exportacao" in df_filtered.columns else None,
                     hover_data=["ano"] if "ano" in df_filtered.columns else None,
                     title="Preço por litro vs Quantidade (por registro)")
    fig.update_layout(xaxis_title="Quantidade (L)", yaxis_title="Valor por litro (US$)", height=520)
    return fig

# ---------------------------
# Insights automáticos (texto)
# ---------------------------

def generate_insights(df_filtered: pd.DataFrame, df_top: pd.DataFrame,
                      climate_by_country: Dict[str, dict], econ_by_country: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Gera insights iniciais automatizados (strings) para apoiar a apresentação.
    Cada insight é breve e baseado em heurísticas simples:
     - total de exportações,
     - principal destino,
     - indicadores climáticos relevantes,
     - PIB per capita dos destinos (quando disponível).
    Observação metodológica (no código): estes insights são preliminares e devem ser validados qualitativamente.
    """
    insights: List[str] = []
    total_valor = float(df_filtered["valor_exportacao"].sum()) if "valor_exportacao" in df_filtered.columns else 0.0
    insights.append(f"Montante acumulado (período selecionado): US$ {total_valor:,.0f}.")
    if not df_top.empty:
        top = df_top.iloc[0]
        insights.append(f"Principal destino: {top['pais']} (~US$ {top['valor_exportacao']:,.0f}).")
    # heurística climática
    for country, clim in climate_by_country.items():
        if clim:
            prec = clim.get("precip_total")
            if prec is not None and prec > 100:
                insights.append(f"{country}: precipitação acumulada recente = {prec:.1f} mm — avaliar impacto logístico e sazonal.")
    # heurística econômica (World Bank)
    for country, econ_df in econ_by_country.items():
        if not econ_df.empty and "value" in econ_df.columns:
            recent = econ_df.dropna().sort_values("year", ascending=False)
            if not recent.empty:
                gdp = recent.iloc[0]["value"]
                insights.append(f"{country}: PIB per capita (último) ~US$ {gdp:,.0f} — considerar segmentação de produto.")
    # recomendação sintética
    insights.append("Recomendação sintética: priorizar mercados com crescimento de PIB per capita e baixa concentração de fornecedores; avaliar investimento em logística para períodos com alta precipitação.")
    return insights

# ---------------------------
# Interface (UI) - renderização com keys únicas
# ---------------------------

def header_ui(kpis: Dict[str, float]):
    """
    Renderiza o cabeçalho com KPIs.
    Comentário acadêmico:
      - KPIs fornecem visão executiva imediata: total de vendas, volume e preço médio.
      - Devem ser o primeiro elemento visto por investidores.
    """
    st.title(APP_TITLE)
    st.markdown("Painel analítico — resumo executivo e seções detalhadas para análise técnica.")
    c1, c2, c3 = st.columns([1.4, 1.4, 1.0])
    # Cada metric tem uma key única para evitar StreamlitDuplicateElementId
    c1.metric("Valor total (US$)", human(kpis["total_valor"]), key="kpi_total_valor")
    c2.metric("Quantidade total (L)", human(kpis["total_litros"]), key="kpi_total_litros")
    c3.metric("Preço médio (US$/L)", human(kpis["preco_medio"]), key="kpi_preco_medio")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    # Sidebar: parâmetros do dashboard
    st.sidebar.header("Configurações")
    csv_path = st.sidebar.text_input("Caminho do CSV local (ou URL)", value=str(DEFAULT_CSV), key="sidebar_csv_path")
    years = st.sidebar.slider("Últimos N anos", min_value=5, max_value=30, value=15, key="sidebar_years")
    top_n = st.sidebar.slider("Top N países", min_value=3, max_value=20, value=10, key="sidebar_top_n")
    climate_months = st.sidebar.number_input("Janela clima (meses)", min_value=1, max_value=36, value=12, key="sidebar_climate_months")
    include_forecast = st.sidebar.checkbox("Incluir forecast linear", value=True, key="sidebar_include_forecast")
    default_forecast_years = 5
    st.sidebar.markdown("---")
    st.sidebar.caption("APIs: REST Countries, Open-Meteo, World Bank — o app possui fallbacks quando ausentes.")

    # Carregar dados
    try:
        df = load_local_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Arquivo CSV não encontrado: {csv_path}")
        st.info("Coloque o CSV em dados_uteis/dados_uteis.csv ou informe caminho/URL válido.")
        return
    except Exception as e:
        logger.exception("Erro lendo CSV: %s", e)
        st.error("Erro ao ler CSV. Ver logs.")
        return

    # Filtrar últimos N anos
    df_filtered = filter_last_n_years(df, int(years))

    # KPIs
    kpis = build_kpis(df_filtered)
    header_ui(kpis)

    # Agregações anuais e top países
    df_year = pd.DataFrame()
    try:
        # Agregação anual (valor e quantidade)
        df_year = df_filtered.groupby("ano", as_index=False).agg({
            "valor_exportacao": "sum" if "valor_exportacao" in df_filtered.columns else pd.NamedAgg(column=None, aggfunc="sum"),
            "quantidade_exportacao": "sum" if "quantidade_exportacao" in df_filtered.columns else pd.NamedAgg(column=None, aggfunc="sum"),
            "valor_exportacao_por_litro": "mean" if "valor_exportacao_por_litro" in df_filtered.columns else pd.NamedAgg(column=None, aggfunc="mean"),
        }).fillna(0)
    except Exception as e:
        logger.warning("Erro na agregação anual: %s", e)
    df_top = top_countries_overall(df_filtered, int(top_n))

    # Abas (mesmo layout do último código)
    tab_geral, tab_valor, tab_quantidade, tab_stats, tab_external, tab_forecast, tab_raw = st.tabs(
        ["Geral", "Valor", "Quantidade", "Estatísticas", "Dados Externos", "Forecast", "Dados Brutos"],
        key="main_tabs"
    )

    # ---------------------------
    # Aba GERAL
    # ---------------------------
    with tab_geral:
        st.subheader("Visão Geral — resumo executivo")
        st.markdown(
            "- Objetivo: apresentar o montante consolidado, identificação dos principais destinos e a tendência dos últimos anos.\n"
            "- Observação metodológica: séries temporais agregadas são sensíveis a eventos extraordinários; investigar anos atípicos separadamente."
        )
        # Evolução do valor (linha)
        fig_val = fig_evolucao_valor(df_year)
        if fig_val:
            st.plotly_chart(fig_val, use_container_width=True, key="chart_valor_geral")
        # Tabela com top países
        st.markdown("**Top países por valor (período selecionado)**")
        if not df_top.empty:
            st.dataframe(df_top.reset_index(drop=True), use_container_width=True, key="df_top_geral")
        else:
            st.info("Sem dados de países para o período selecionado.")

    # ---------------------------
    # Aba VALOR
    # ---------------------------
    with tab_valor:
        st.subheader("Análise detalhada por Valor (US$)")
        st.markdown("Nesta seção examinamos exclusivamente as métricas monetárias e sua distribuição por destino.")
        # Reutiliza fig_val mas com key distinto
        if fig_val:
            st.plotly_chart(fig_val, use_container_width=True, key="chart_valor_detalhado")
            st.markdown("**Interpretação (nota no código):** verificar se o aumento do valor decorre de volume ou de preço por litro comparando com 'Quantidade' e 'Estatísticas'.")
        # Top países (barra)
        fig_top = fig_top_paises_bar(df_top)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True, key="chart_top_paises_valor")
        # Detalhe por país (tabela)
        st.markdown("**Detalhamento por país (valor e quantidade)**")
        if not df_top.empty:
            st.dataframe(df_top.rename(columns={"valor_exportacao": "Valor (US$)", "quantidade_exportacao": "Quantidade (L)"}), use_container_width=True, key="df_top_valor")
        else:
            st.info("Sem dados para exibir tabela de países.")

    # ---------------------------
    # Aba QUANTIDADE
    # ---------------------------
    with tab_quantidade:
        st.subheader("Análise detalhada por Quantidade (L)")
        st.markdown("Foco no volume exportado — útil para entender capacidade produtiva e sazonalidade.")
        fig_qtd = fig_evolucao_quantidade(df_year)
        if fig_qtd:
            st.plotly_chart(fig_qtd, use_container_width=True, key="chart_quantidade_evolucao")
            st.markdown("**Interpretação (nota no código):** divergência entre valor e quantidade sugere alteração de preço médio.")
        # Scatter preço x volume (auxiliar)
        fig_scatter = fig_price_vs_volume_scatter(df_filtered)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True, key="chart_scatter_preco_volume")

    # ---------------------------
    # Aba ESTATÍSTICAS
    # ---------------------------
    with tab_stats:
        st.subheader("Estatísticas e indicadores complementares")
        st.markdown("Apresentamos métricas que ajudam a decompor o efeito preço vs volume e analisar percentuais de exportação.")
        fig_price = fig_preco_medio_ano(df_filtered)
        if fig_price:
            st.plotly_chart(fig_price, use_container_width=True, key="chart_preco_medio")
            st.markdown("**Nota metodológica (no código):** preço médio por litro pode ser calculado explicitamente ou inferido pelo total/volume agregado.")
        # Percentual exportado (se disponível)
        if "percentual_exportacao" in df_filtered.columns:
            pct_by_year = df_filtered.groupby("ano", as_index=False)["percentual_exportacao"].mean()
            fig_pct = px.line(pct_by_year, x="ano", y="percentual_exportacao", markers=True, title="Percentual médio da produção exportado (%)")
            fig_pct.update_layout(yaxis_title="Percentual (%)", xaxis_title="Ano", height=420)
            st.plotly_chart(fig_pct, use_container_width=True, key="chart_percentual_exportacao")
        else:
            st.info("Coluna 'percentual_exportacao' ausente no CSV. Se disponível, será exibida aqui.")

    # ---------------------------
    # Aba DADOS EXTERNOS
    # ---------------------------
    with tab_external:
        st.subheader("Dados externos por Top Países (Clima, Economia, Reviews proxy)")
        st.markdown("Integração com REST Countries, Open-Meteo e World Bank (PIB per capita). Quando dados ausentes, apresentamos 'N/D'.")
        climate_by_country: Dict[str, dict] = {}
        econ_by_country: Dict[str, pd.DataFrame] = {}
        today = datetime.utcnow().date()
        start_date = (today - timedelta(days=30 * int(climate_months))).isoformat()
        end_date = today.isoformat()

        if not df_top.empty:
            top_list = df_top["pais"].tolist()
            # Progress indicator com key único
            prog = st.progress(0, key="progress_external")
            for i, country in enumerate(top_list):
                st.markdown(f"#### {country}")
                info = get_country_info(country)
                lat, lon, iso2 = None, None, ""
                if info:
                    latlng = safe_get(info, "latlng", [])
                    if isinstance(latlng, list) and len(latlng) >= 2:
                        lat, lon = latlng[0], latlng[1]
                    cca2 = safe_get(info, "cca2", "")
                    iso2 = cca2[:2].lower() if cca2 else ""
                # climate
                clim = get_climate_summary(lat, lon, start_date, end_date) if lat and lon else {}
                climate_by_country[country] = clim
                # econ
                econ_df = get_worldbank_indicator(iso2, start=today.year - int(years), end=today.year) if iso2 else pd.DataFrame()
                econ_by_country[country] = econ_df
                c1, c2, c3 = st.columns(3)
                c1.metric("Temp max média (°C)", clim.get("temp_max_avg") or "N/D", key=f"metric_temp_{i}")
                c2.metric("Precipitação total (mm)", clim.get("precip_total") or "N/D", key=f"metric_precip_{i}")
                # tratamento seguro do WorldBank DF
                if not econ_df.empty and "year" in econ_df.columns and "value" in econ_df.columns:
                    recent = econ_df.dropna().sort_values("year", ascending=False).head(1)
                    gdp_display = f"US$ {int(recent.iloc[0]['value']):,}" if not recent.empty else "N/D"
                else:
                    gdp_display = "N/D"
                c3.metric("PIB per capita (último)", gdp_display, key=f"metric_gdp_{i}")
                # reviews proxy
                wrev = get_wine_review_proxy(country)
                st.caption(f"Avaliação média proxy: **{wrev['avg_score']}** (n={wrev['reviews_count']})", key=f"caption_rev_{i}")
                prog.progress(int((i + 1) / len(top_list) * 100))
            prog.empty()
        else:
            st.info("Top países não calculados — verifique filtros e CSV.")

    # ---------------------------
    # Aba FORECAST
    # ---------------------------
    with tab_forecast:
        st.subheader("Forecast linear (exploratório)")
        st.markdown("Método: regressão linear anual. Usar apenas para cenários exploratórios; validar com modelos avançados para decisões.")
        # Preparar série anual
        if "valor_exportacao" in df_filtered.columns:
            df_series = df_filtered.groupby("ano", as_index=False)["valor_exportacao"].sum().rename(columns={"valor_exportacao": "value"})
        else:
            df_series = pd.DataFrame()
        if df_series.empty or len(df_series) < 2:
            st.info("Dados insuficientes para gerar forecast (mínimo 2 anos com valor).")
        else:
            # Campo para escolher anos de forecast (default 5)
            n_future = st.number_input("Anos futuros para prever (linear)", min_value=1, max_value=10, value=default_forecast_years, key="input_forecast_years")
            df_pred = simple_linear_forecast(df_series.rename(columns={"value": "value"}), n_future=int(n_future))
            if not df_pred.empty:
                df_plot = pd.concat([df_series.rename(columns={"value": "value"}), df_pred.rename(columns={"value": "value"})], ignore_index=True)
                fig_forecast = px.line(df_plot, x="ano", y="value", markers=True, title="Histórico + Forecast (linear) — Valor exportado (US$)")
                fig_forecast.add_vline(x=int(df_series["ano"].max()), line_dash="dash", line_color="gray")
                st.plotly_chart(fig_forecast, use_container_width=True, key="chart_forecast")
                st.markdown("**Nota acadêmica:** o modelo linear é preliminar. Para robustez, realizar validação temporal e usar modelos estatísticos sofisticados (Prophet, ARIMA, ETS).")

    # ---------------------------
    # Aba DADOS BRUTOS
    # ---------------------------
    with tab_raw:
        st.subheader("Dados brutos — versão filtrada")
        st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True, key="df_raw")
        st.download_button("Baixar CSV filtrado", df_filtered.to_csv(index=False).encode("utf-8"),
                           "exportacoes_filtradas.csv", "text/csv", key="download_filtered")

    # ---------------------------
    # Insights automáticos (sidebar)
    # ---------------------------
    st.sidebar.markdown("---")
    if st.sidebar.button("Gerar insights automáticos", key="btn_generate_insights"):
        with st.spinner("Gerando insights..."):
            insights = generate_insights(df_filtered, df_top,
                                         climate_by_country if 'climate_by_country' in locals() else {},
                                         econ_by_country if 'econ_by_country' in locals() else {})
            st.sidebar.markdown("### Insights automáticos")
            for idx, it in enumerate(insights, 1):
                st.sidebar.write(f"{idx}. {it}")

    # Rodapé / nota
    st.caption("V4 Analítico — integra dados externos (Open-Meteo, REST Countries, World Bank). Método de forecast linear é exploratório. Desenvolvido para Tech Challenge - Fase 1.")

if __name__ == "__main__":
    main()
