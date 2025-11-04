"""
app_v5_exec_final_offline_continent.py
Dashboard Executivo e Analítico de Exportações de Vinho

Principais ajustes:
- Treemap offline (CONTINENT_MAP) para eliminar 'Outros' indevido.
- Fallback de coordenadas usando `capitalInfo.latlng` para reduzir 'N/D' no contexto climático.
- Correção de st.progress() (sem key).
- Comentários em português e cache para chamadas externas.
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
logger = logging.getLogger("app_v5_exe
