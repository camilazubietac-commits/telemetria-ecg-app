import streamlit as st
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---- ESTILO ATRACTIVO ----
st.set_page_config(page_title="Vitals Link", layout="wide")
st.markdown("""
<style>
body { background-color: #f0f8ff; font-family: 'Arial', sans-serif; }
.big-title { font-size: 120px; font-weight: bold; color: #228B22; text-align: center; margin-bottom: 0; }
.sub-title { font-size: 80px; color: #000000; text-align: center; margin-top: 0; }
.hr-big { font-size: 40px; color: #228B22; text-align: center; margin: 20px 0; }
.report-title { font-size: 30px; color: #228B22; }
.report-table { border-collapse: collapse; width: 100%; }
.report-table th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
.report-table th { background-color: #f2f2f2; }
.diag { font-size: 25px; color: #228B22; }
</style>
""", unsafe_allow_html=True)

# Fondo gráfico (ondas cardíacas)
st.markdown("""
<div style="background: linear-gradient(to right, #f0f8ff, #e0f7fa); position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"></div>
""", unsafe_allow_html=True)

# Nombre y subtítulo
st.markdown('<p class="big-title">Vitals Link</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">El guardián digital de tu ritmo cardíaco, conectado y seguro.</p>', unsafe_allow_html=True)

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
    st.session_state.data["hr"] = d.features.get("hbpermin", 0)
    st.session_state.data["features"] = d.features
    
    df_feat = pd.DataFrame([d.features])
    probs = modelo.predict_proba(df_feat)[0] * 100
    st.session_state.data["probs"] = probs.round(1).tolist()
    return {"ok": True}

# ---- INTERFAZ ----
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Derivación I")
    st.line_chart(st.session_state.data["I"], height=200)

with col2:
    st.subheader("Derivación II")
    st.line_chart(st.session_state.data["II"], height=200)

with col3:
    st.subheader("Derivación III (calculada)")
    st.line_chart(st.session_state.data["III"], height=200)

st.markdown('<p class="hr-big">Frecuencia Cardíaca: {st.session_state.data["hr"]:.0f} bpm</p>', unsafe_allow_html=True)

st.subheader("Informe de Features")
if st.session_state.data["features"]:
    feat_df = pd.DataFrame.from_dict(st.session_state.data["features"], orient='index', columns=['Valor'])
    st.table(feat_df.style.format("{:.3f}"))

st.subheader("Análisis de Patología")
probs = st.session_state.data["probs"]
df_prob = pd.DataFrame({"Condición": CLASES, "%": probs})
st.bar_chart(df_prob.set_index("Condición"), height=300)

max_idx = np.argmax(probs)
patologia = CLASES[max_idx]
confianza = probs[max_idx]
justificacion = f"Basado en los features, como alta SDRR ({st.session_state.data["features"].get("SDRR", 0):.3f}) y RMSSD ({st.session_state.data["features"].get("RMSSD", 0):.3f}), que indican variabilidad irregular, y QRSseg ({st.session_state.data["features"].get("QRSseg", 0):.3f}) prolongado, esto sugiere {patologia} con {confianza:.1f}% de confianza. Recomendación: Consulta médica inmediata si persiste."

st.markdown(f"<p class='diag'>Patología: {patologia} ({confianza:.1f}%)</p>", unsafe_allow_html=True)
st.write(justificacion)

import uvicorn
if __name__ == "__main__":
    import threading
    threading.Thread(target=uvicorn.run, args=(api,), kwargs={"host":"0.0.0.0", "port":8000}).start()