import streamlit as st
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ==================== CONFIGURACIÓN INICIAL ====================
st.set_page_config(page_title="Vitals Link", layout="wide")

# CSS
st.markdown("""
<style>
    .big-title {font-size: 80px !important; font-weight: bold; text-align: center; margin-bottom: 0px;}
    .sub-title {font-size: 28px !important; text-align: center; margin-top: 5px; font-weight: normal; opacity: 0.9;}
    .hr-big {font-size: 70px !important; text-align: center; font-weight: bold;}
    body {background: linear-gradient(135deg, #e8f5e9, #f1f8e8);}
</style>
""", unsafe_allow_html=True)

# ==================== ESTADO INICIAL SEGURO ====================
if "data" not in st.session_state:
    st.session_state.data = {
        "I": [0]*600,
        "II": [0]*600,
        "III": [0]*600,
        "hr": 0,
        "features": {},
        "probs": [25, 25, 25, 25]  # valor por defecto
    }

# ==================== COLOR SEGÚN DIAGNÓSTICO (sin errores) ====================
idx = np.argmax(st.session_state.data["probs"])
colores = ["#d32f2f", "#f57c00", "#c62828", "#2e7d32"]  # AFF, ARR, CHF, NSR
color_actual = colores[idx]

# Título y subtítulo (ahora sí funciona desde el segundo 1)
st.markdown(f'<h1 class="big-title" style="color:{color_actual};">Vitals Link</h1>', unsafe_allow_html=True)
st.markdown(f'<h3 class="sub-title" style="color:{color_actual};">El guardián digital de tu ritmo cardíaco, conectado y seguro.</h3>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==================== CARGA DEL MODELO ====================
modelo = joblib.load("modelo_ecg_random_forest_final.pkl")
CLASES = ['Fibrilación Auricular', 'Arritmia', 'Insuficiencia Cardíaca', 'Ritmo Sinusal Normal']

# ==================== API ====================
api = FastAPI()

class DatosECG(BaseModel):
    derivacion_I: list
    derivacion_II: list
    derivacion_III: list
    features: dict

@api.post("/predict")
def recibir(d: DatosECG):
    st.session_state.data["I"] = d.derivacion_I[-600:]
    st.session_state.data["II"] = d.derivacion_II[-600:]
    st.session_state.data["III"] = d.derivacion_III[-600:]
    st.session_state.data["hr"] = round(d.features.get("hbpermin", 0))
    st.session_state.data["features"] = d.features
    
    df_feat = pd.DataFrame([d.features])
    probs = modelo.predict_proba(df_feat)[0] * 100
    st.session_state.data["probs"] = probs.round(1).tolist()
    return {"status": "ok"}

# ==================== INTERFAZ ====================
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Derivación I")
    st.line_chart(st.session_state.data["I"], height=220, use_container_width=True)
with col2:
    st.subheader("Derivación II")
    st.line_chart(st.session_state.data["II"], height=220, use_container_width=True)
with col3:
    st.subheader("Derivación III")
    st.line_chart(st.session_state.data["III"], height=220, use_container_width=True)

st.markdown(f'<p class="hr-big" style="color:{color_actual};">{st.session_state.data["hr"]} <small style="font-size:50px;">bpm</small></p>', unsafe_allow_html=True)

st.subheader("12 Features Extraídos")
if st.session_state.data["features"]:
    feat_df = pd.DataFrame.from_dict(st.session_state.data["features"], orient='index', columns=['Valor'])
    st.dataframe(feat_df.style.format("{:.4f}"), use_container_width=True)
else:
    st.info("Esperando señal del ESP32...")

st.subheader("Diagnóstico Automático")
probs = st.session_state.data["probs"]
idx = np.argmax(probs)
patologia = CLASES[idx]
confianza = probs[idx]

colA, colB = st.columns([2, 3])
with colA:
    st.markdown(f"<h2 style='color:{color_actual}; text-align:center;'>{patologia}<br>{confianza:.1f}%</h2>", unsafe_allow_html=True)
with colB:
    st.write(f"Frecuencia: {st.session_state.data['hr']} bpm | QRS: {st.session_state.data['features'].get('QRSseg',0):.3f}s | SDRR: {st.session_state.data['features'].get('SDRR',0):.3f}")

# ==================== SERVIDOR ====================
import uvicorn
if __name__ == "__main__":
    import threading
    threading.Thread(target=uvicorn.run, args=(api,), kwargs={"host":"0.0.0.0", "port":8000}).start()