import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from core.utils import ScenarioInputs, ScenarioResults
from core.finance import build_cash_flows, npv_irr_payback, lcox_like, monte_carlo, breakeven_price
from core.carbon import annual_carbon, totals
from core.optimize import optimize_config

st.set_page_config(page_title='AI-Enhanced Investment – Energy Transition', layout='wide')

# lazy init helper
def get_policy_rag():
    if 'policy_rag' in st.session_state:
        return st.session_state.policy_rag
    try:
        from core.policy import PolicyRAG
        st.session_state.policy_rag = PolicyRAG()
        return st.session_state.policy_rag
    except Exception as e:
        # store flag to avoid repeated failing imports
        st.session_state.policy_rag = None
        st.session_state.policy_rag_error = str(e)
        return None

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Upload & Assumptions', 'Scenario Results', 'Sensitivity Dashboard', 'Policy Explorer', 'Cases Library'])

@st.cache_data
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Page 1: Upload & Assumptions ---
if page == 'Upload & Assumptions':
    st.header('Upload & Assumptions')

    st.subheader('Project Core Inputs')
    col1, col2, col3 = st.columns(3)
    with col1:
        years = st.number_input('Project Life (years)', 5, 40, 20)
        capex = st.number_input('Initial CapEx (USD)', min_value=0.0, value=200_000_000.0, step=1e6, format='%.0f')
        opex = st.number_input('Annual OpEx (USD/yr)', min_value=0.0, value=15_000_000.0, step=1e5, format='%.0f')
        discount_rate = st.number_input('Discount Rate / WACC', min_value=0.0, max_value=0.5, value=0.1, step=0.005, format='%.3f')
    with col2:
        capacity = st.number_input('Production/Throughput (units/yr)', min_value=0.0, value=1_000_000.0, step=10_000.0)
        price = st.number_input('Product Price (USD/unit)', min_value=0.0, value=80.0, step=1.0)
        availability = st.slider('Availability', 0.5, 0.99, 0.9)
        degradation = st.slider('Annual Degradation', 0.0, 0.05, 0.0)
    with col3:
        inflation = st.slider('Opex Inflation', 0.0, 0.15, 0.02)
        salvage = st.number_input('Salvage Value (USD at end)', min_value=0.0, value=0.0, step=1e6, format='%.0f')
        construction = st.number_input('Construction Years', 1, 5, 1)

    st.subheader('Carbon & Policy Inputs')
    col4, col5, col6 = st.columns(3)
    with col4:
        baseline_i = st.number_input('Baseline Emissions Intensity (tCO2e/unit)', 0.0, 10.0, 0.8)
        project_i = st.number_input('Project Emissions Intensity (tCO2e/unit)', 0.0, 10.0, 0.2)
    with col5:
        carbon_price = st.number_input('Carbon Price (USD/tCO2e, Year 1)', 0.0, 500.0, 40.0)
        carbon_growth = st.slider('Carbon Price Annual Growth', 0.0, 0.5, 0.05)
    with col6:
        st.info('Upload policy PDFs below to enable RAG on the Policy Explorer tab.')

    # Build input object
    p = ScenarioInputs(
        years=int(years), discount_rate=float(discount_rate), capex_initial=float(capex), opex_annual=float(opex),
        capacity_per_year=float(capacity), product_price=float(price), availability=float(availability),
        degradation=float(degradation), inflation=float(inflation), salvage_value=float(salvage),
        construction_years=int(construction), baseline_intensity=float(baseline_i), project_intensity=float(project_i),
        carbon_price=float(carbon_price), carbon_price_growth=float(carbon_growth)
    )

    st.session_state.inputs = p
    st.success('Inputs captured. Move to Scenario Results to compute metrics.')

    st.subheader('Upload Policy/ITB/SoW PDFs (optional)')
    f = st.file_uploader('Upload PDF files', type=['pdf'], accept_multiple_files=True)
    if f:
        rag = get_policy_rag()
        if rag is None:
            st.error("Policy RAG is unavailable: " + st.session_state.get('policy_rag_error', 'unknown error') + 
                     " — install DuckDB backend or update sqlite. See README.")
        else:
            for uploaded in f:
                bytes_data = uploaded.read()
                tmp_path = os.path.join('/tmp', uploaded.name)
                with open(tmp_path, 'wb') as fh:
                    fh.write(bytes_data)
                rag.add_pdf(tmp_path, meta={'tag': 'uploaded'})
            st.success(f'Indexed {len(f)} file(s) into the policy knowledge base.')

# --- Page 2: Scenario Results ---
elif page == 'Scenario Results':
    st.header('Scenario Results')
    if 'inputs' not in st.session_state:
        st.warning('Please set inputs on the Upload & Assumptions page first.')
        st.stop()
    p = st.session_state.inputs

    cash_flows, annual = build_cash_flows(p)
    npv, irr, payback = npv_irr_payback(cash_flows, p.discount_rate)
    lcox = lcox_like(p, cash_flows)
    co2_rows = annual_carbon(p)
    co2_total, carbon_credits_total = totals(co2_rows)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('NPV (USD)', f"{npv:,.0f}")
    col2.metric('IRR', f"{irr*100:.2f}%" if np.isfinite(irr) else 'n/a')
    col3.metric('Payback (yrs)', f"{payback:.2f}" if np.isfinite(payback) else 'n/a')
    col4.metric('Levelized Cost (proxy)', f"${lcox:,.2f}/unit")

    col5, col6 = st.columns(2)
    col5.metric('CO₂e Avoided (t over life)', f"{co2_total:,.0f}")
    col6.metric('Carbon Credit Value (USD)', f"{carbon_credits_total:,.0f}")

    df_fin = pd.DataFrame(annual)
    df_carbon = pd.DataFrame(co2_rows)
    st.subheader('Annual Cash Flow Summary')
    st.dataframe(df_fin, use_container_width=True)
    fig1 = px.bar(df_fin, x='year', y=['revenue','opex','cash_flow'], title='Revenue, Opex, Cash Flow')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader('Annual Emissions & Credits')
    st.dataframe(df_carbon, use_container_width=True)
    fig2 = px.line(df_carbon, x='year', y=['baseline_tCO2e','project_tCO2e','avoided_tCO2e'], title='Emissions Path')
    st.plotly_chart(fig2, use_container_width=True)

    # Breakeven price
    be_price = breakeven_price(p)
    st.info(f"Breakeven Product Price for NPV=0: ${be_price:,.2f} per unit")

    # Export
    with st.expander('Download Results'):
        st.download_button('Annual Financials CSV', df_to_csv_bytes(df_fin), 'annual_financials.csv', 'text/csv')
        st.download_button('Annual Carbon CSV', df_to_csv_bytes(df_carbon), 'annual_carbon.csv', 'text/csv')

# --- Page 3: Sensitivity Dashboard ---
elif page == 'Sensitivity Dashboard':
    st.header('Sensitivity & Risk (Monte Carlo)')
    if 'inputs' not in st.session_state:
        st.warning('Please set inputs first.')
        st.stop()
    p = st.session_state.inputs

    col1, col2, col3, col4 = st.columns(4)
    price_sigma = col1.slider('Price Volatility σ (lognormal)', 0.0, 0.5, 0.15)
    capex_sigma = col2.slider('CapEx Volatility σ', 0.0, 0.5, 0.15)
    opex_sigma  = col3.slider('OpEx Volatility σ', 0.0, 0.5, 0.10)
    rate_sigma  = col4.slider('WACC StdDev', 0.0, 0.2, 0.02)

    runs = st.slider('Monte Carlo runs', 100, 5000, 1000, step=100)
    with st.spinner('Running Monte Carlo...'):
        npvs = monte_carlo(p, n=runs, price_sigma=price_sigma, capex_sigma=capex_sigma, opex_sigma=opex_sigma, rate_sigma=rate_sigma)
    st.write(f"Mean NPV: ${np.mean(npvs):,.0f} | P(NPV>0): {100*np.mean(npvs>0):.1f}%")
    fig = px.histogram(npvs, nbins=50, title='NPV Distribution')
    st.plotly_chart(fig, use_container_width=True)

# --- Page 4: Policy Explorer ---
elif page == 'Policy Explorer':
    st.header('Policy & Geopolitics Explorer (RAG)')
    st.write('Ask questions about your uploaded policy PDFs / ITB / SOW documents. The system retrieves the most relevant passages and runs sentiment on pasted news.')

    q = st.text_input('Query')
    if st.button('Search') and q:
        rag = get_policy_rag()
        if rag is None:
            st.error("Policy RAG is unavailable: " + st.session_state.get('policy_rag_error', 'unknown error') + 
                     " — install DuckDB backend or update sqlite. See README.")
        else:
            hits = rag.query(q, k=5)
            for h in hits:
                with st.expander(f"{h['meta'].get('source')} – page {h['meta'].get('page')} (distance {h['distance']:.3f})"):
                    st.write(h['text'][:2000])

    st.subheader('News / Headlines Sentiment (optional)')
    news = st.text_area('Paste recent headlines or policy notes')
    if st.button('Analyze Sentiment') and news:
        rag = get_policy_rag()
        if rag is None:
            st.error("Policy RAG is unavailable: " + st.session_state.get('policy_rag_error', 'unknown error') + 
                     " — install DuckDB backend or update sqlite. See README.")
        else:
            s = rag.sentiment(news)
            st.json(s)

# --- Page 5: Cases Library ---
else:
    st.header('Cases Library')
    st.write('Load a sample green ammonia case with pre-filled inputs (hypothetical).')
    if st.button('Load Sample Case'):
        st.session_state.inputs = ScenarioInputs(
            years=20, discount_rate=0.10, capex_initial=350_000_000, opex_annual=22_000_000,
            capacity_per_year=650_000, product_price=420, availability=0.9, degradation=0.00,
            inflation=0.02, salvage_value=0.0, construction_years=2,
            baseline_intensity=1.8, project_intensity=0.2, carbon_price=60, carbon_price_growth=0.05
        )
        st.success('Sample case loaded. Go to Scenario Results.')

    st.subheader('Simple Optimization (Pareto-style)')
    if 'inputs' in st.session_state and st.button('Run Optimization'):
        best, combos = optimize_config(st.session_state.inputs, weights=(0.5,0.5))
        st.write('Best configuration:', best)
        df = pd.DataFrame(combos)
        fig = px.scatter(df, x='co2', y='npv', color=df['wacc'].astype(str), symbol=df['scale'].astype(str), title='Trade-off: tCO₂e avoided vs NPV')
        st.plotly_chart(fig, use_container_width=True)

import os
from typing import List, Dict
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class PolicyRAG:
    def __init__(self, persist_dir: str = '.chromadb'):
        # Use DuckDB+Parquet backend to avoid requiring a modern system sqlite3
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            persist_directory=persist_dir,
            chroma_db_impl="duckdb+parquet"
        ))
        self.collection = self.client.get_or_create_collection(name='policies')
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.sa = SentimentIntensityAnalyzer()

    def add_pdf(self, filepath: str, meta: Dict = None):
        # ... existing code ...
