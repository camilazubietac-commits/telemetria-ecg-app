import streamlit as st
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---- ESTILO BONITO ----
st.set_page_config(page_title="Telemetría ECG", layout="wide")
st.markdown("""
<style>
.big-font {font-size:50px !important; font-weight: bold; color: #1E88E5;}
.medium-font {font-size:25px !important;}
.css-1d391kg {padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🫀 Telemetría ECG en Tiempo Real")
st.markdown("**Camila Zubieta** • Actualización cada 6 segundos • Abre desde tu celular")

# Cargar modelo
modelo = joblib.load("modelo_ecg_random_forest_final.pkl")
CLASES = ['Fibrilación (AFF)', 'Arritmia (ARR)', 'Insuficiencia (CHF)', 'Normal (NSR)']

# Variables globales
if "data" not in st.session_state:
    st.session_state.data = {"I": [0]*600, "II": [0]*600, "III": [0]*600, 
                            "hr": 0, "features": {}, "probs": [25,25,25,25]}

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
    st.session_state.data["features"] = d.features
    st.session_state.data["hr"] = d.features.get("hbpermin", 0)
    
    df_feat = pd.DataFrame([d.features])
    probs = modelo.predict_proba(df_feat)[0] * 100
    st.session_state.data["probs"] = probs.round(1).tolist()
    return {"ok": True}

# ---- INTERFAZ ----
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Derivaciones ECG")
    st.line_chart({
        "Derivación I": st.session_state.data["I"],
        "Derivación II": st.session_state.data["II"],
        "Derivación III": st.session_state.data["III"]
    }, height=400)

with col2:
    st.markdown(f'<p class="big-font">{st.session_state.data["hr"]:.0f} <small>bpm</small></p>', 
                unsafe_allow_html=True)
    st.markdown("**Frecuencia Cardíaca**")

    probs = st.session_state.data["probs"]
    df_prob = pd.DataFrame({"Condición": CLASES, "%": probs})
    st.bar_chart(df_prob.set_index("Condición"), height=300)
    
    max_idx = np.argmax(probs)
    color = ["⚠️", "⚠️", "⚠️", "✅"][max_idx]
    st.markdown(f"### {color} **DIAGNÓSTICO:** {CLASES[max_idx]}")
    st.markdown(f"**Confianza:** {probs[max_idx]:.1f}%")

st.subheader("12 Features evaluados")
if st.session_state.data["features"]:
    feat_df = pd.DataFrame([st.session_state.data["features"]]).T
    feat_df.columns = ["Valor"]
    st.table(feat_df.style.format("{:.3f}"))

import uvicorn
if __name__ == "__main__":
    import threading
    threading.Thread(target=uvicorn.run, args=(api,), kwargs={"host":"0.0.0.0", "port":8000}).start()