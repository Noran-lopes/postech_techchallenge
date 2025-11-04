# 📊 Tech Challenge — Fase 1  
**Pós-Tech Data Analytics — FIAP / Alura**

## 🎯 Objetivo
Desenvolver um **dashboard analítico e interativo** em Python (usando Streamlit) que apresente as **exportações brasileiras de vinho** nos últimos anos, explorando:
- Valor exportado (US$)
- Quantidade exportada (litros)
- Preço médio por litro
- Percentual exportado da produção (se disponível)
- Dados externos: **clima**, **economia** e **avaliações**
- **Previsões (Forecast)** de tendência simples

---

## 🧠 Descrição Técnica
O dashboard foi desenvolvido para **atender integralmente o desafio proposto**:
- Utiliza dados da **Embrapa / Vitibrasil**
- Analisa **os últimos N anos (configurável)**
- Apresenta **insights e gráficos explicativos**
- Integra **APIs externas**:
  - 🌎 REST Countries → informações geográficas (lat/lon, ISO)
  - ☀️ Open-Meteo → dados de temperatura e precipitação
  - 💰 World Bank → PIB per capita (indicador NY.GDP.PCAP.CD)
- Gera **forecast linear simples** (exploratório)
- Cria **insights automáticos** e recomendações baseadas em dados

---

## 🧰 Tecnologias Utilizadas
| Tecnologia | Uso principal |
|-------------|----------------|
| **Python 3.10+** | Linguagem principal |
| **Streamlit** | Framework web para dashboards |
| **Pandas** | Manipulação e análise de dados |
| **Plotly** | Gráficos interativos |
| **NumPy** | Cálculos estatísticos e previsão linear |
| **Requests** | Consumo de APIs REST externas |

---

## ⚙️ Como executar localmente

1. **Clone o repositório** ou copie os arquivos:
   ```bash
   git clone https://github.com/seuusuario/postech_techchallenge.git
   cd postech_techchallenge
