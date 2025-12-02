import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# ==================== CARGA DEL MODELO ====================
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_ecg_random_forest_final.pkl")

modelo = cargar_modelo()
CLASES = ['Fibrilación Auricular', 'Arritmia', 'Insuficiencia Cardíaca', 'Ritmo Sinusal Normal']

# ==================== CONFIGURACIÓN ====================
st.set_page_config(page_title="Vitals Link", layout="wide")
st.markdown("""
<style>
    .big-title {font-size: 80px !important; font-weight: bold; text-align: center;}
    .sub-title {font-size: 28px !important; text-align: center; opacity: 0.9;}
    .hr-big {font-size: 70px !important; text-align: center; font-weight: bold;}
    body {background: linear-gradient(135deg, #e8f5e9, #f1f8e8);}
</style>
""", unsafe_allow_html=True)

# ==================== ESTADO INICIAL ====================
if "data" not in st.session_state:
    st.session_state.data = {
        "I": [0]*600, "II": [0]*600, "III": [0]*600,
        "hr": 0, "features": {}, "probs": [25,25,25,25]
    }

# ==================== RECEPTOR UNIVERSAL (GET + POST) ====================
query = st.query_params

# 1. Recibe BPM directo
if "hbpermin" in query:
    st.session_state.data["hr"] = int(query["hbpermin"])

# 2. Recibe JSON completo (derivaciones + features)
if "data" in query:
    try:
        data = json.loads(query["data"])
        st.session_state.data["I"]   = data.get("derivacion_I",  st.session_state.data["I"])[:600]
        st.session_state.data["II"]  = data.get("derivacion_II", st.session_state.data["II"])[:600]
        st.session_state.data["III"] = data.get("derivacion_III",st.session_state.data["III"])[:600]
        if "features" in data:
            st.session_state.data["features"] = data["features"]
            st.session_state.data["hr"] = round(data["features"].get("hbpermin", st.session_state.data["hr"]))
            df_feat = pd.DataFrame([data["features"]])
            probs = modelo.predict_proba(df_feat)[0] * 100
            st.session_state.data["probs"] = probs.round(1).tolist()
        st.rerun()
    except:
        pass

# ==================== INTERFAZ ====================
idx = np.argmax(st.session_state.data["probs"])
colores = ["#d32f2f", "#f57c00", "#c62828", "#2e7d32"]
color_actual = colores[idx]

st.markdown(f'<h1 class="big-title" style="color:{color_actual};">Vitals Link</h1>', unsafe_allow_html=True)
st.markdown(f'<h3 class="sub-title" style="color:{color_actual};">Monitoreo cardíaco en tiempo real</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1: st.subheader("Derivación I");  st.line_chart(st.session_state.data["I"], height=220)
with col2: st.subheader("Derivación II"); st.line_chart(st.session_state.data["II"], height=220)
with col3: st.subheader("Derivación III");st.line_chart(st.session_state.data["III"], height=220)

st.markdown(f'<p class="hr-big" style="color:{color_actual};">{st.session_state.data["hr"]} <small>bpm</small></p>', unsafe_allow_html=True)

if st.session_state.data["features"]:
    df = pd.DataFrame.from_dict(st.session_state.data["features"], orient='index', columns=['Valor'])
    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

patologia = CLASES[idx]
confianza = st.session_state.data["probs"][idx]
st.markdown(f"<h2 style='color:{color_actual}; text-align:center;'>{patologia} – {confianza:.1f}%</h2>", unsafe_allow_html=True)