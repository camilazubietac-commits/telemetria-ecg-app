import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ←←←←← El modelo ML está aquí: carga tu archivo .pkl (que copiarás en la carpeta)
modelo = joblib.load("modelo_ecg_random_forest_final.pkl")

st.set_page_config(page_title="ECG EN VIVO", layout="wide")
st.title("ECG + Diagnóstico en Tiempo Real 🩺")
st.write("Abre esta página desde tu celular. Actualiza cada 5-7 segundos.")

# Variables que se actualizan cuando llegan datos del ESP32
if "ecg" not in st.session_state:
    st.session_state.ecg = [0] * 1000  # Buffer para la señal ECG
    st.session_state.hr = 0  # Frecuencia cardíaca inicial
    st.session_state.probs = [25, 25, 25, 25]  # Probabilidades iniciales

# Parte de recepción de datos (FastAPI dentro de Streamlit)
from fastapi import FastAPI
from pydantic import BaseModel

api = FastAPI()

class DatosECG(BaseModel):
    ecg: list  # El array de la señal ECG del ESP32

@api.post("/predict")
def recibir(datos: DatosECG):
    # Guardar la señal recibida
    st.session_state.ecg = datos.ecg[-1000:]  # Últimos 1000 valores

    # Calcular frecuencia cardíaca simple (conteo de cruces por umbral)
    señal = np.array(datos.ecg)
    cruces = len(np.where((señal[:-1] < 1.65) & (señal[1:] > 1.65))[0])
    st.session_state.hr = round(cruces * 12)  # 5 segundos de datos → x12 para bpm

    # Features mínimas para el modelo ML (esto se conecta con tu CSV de training)
    # Aquí usamos valores placeholders; mañana lo mejoramos con features reales de la señal
    features = {
        'hbpermin': max(st.session_state.hr, 40),  # Frecuencia cardíaca
        'RRmean': 60000 / max(st.session_state.hr, 40),  # Media de intervalos RR
        'SDRR': 60, 'RMSSD': 45, 'pNN50': 8,  # Variabilidad
        'QRSseg': 0.1, 'QTseg': 0.38, 'PRseg': 0.16,  # Duraciones
        'Pseg': 0.11, 'Tseg': 0.22, 'QRSarea': 1.1  # Amplitudes y áreas
    }

    # Ejecutar el modelo ML
    df = pd.DataFrame([features])
    prob = modelo.predict_proba(df)[0] * 100  # Calcula % para cada clase
    clases = ['Fibrilación (AFF)', 'Arritmia (ARR)', 'Insuf. Cardíaca (CHF)', 'Normal (NSR)']

    st.session_state.probs = prob.round(1).tolist()  # Guardar probabilidades
    return {"status": "datos recibidos OK"}

# === Interfaz gráfica (lo que ves en el navegador) ===
col1, col2 = st.columns([2,1])

with col1:
    st.line_chart(st.session_state.ecg, height=350)
    st.caption("Señal ECG en tiempo real (onda del corazón)")

with col2:
    st.metric("Frecuencia Cardíaca", f"{st.session_state.hr} bpm")

    if sum(st.session_state.probs) > 0:
        df_bar = pd.DataFrame({
            "Condición": ['Fibrilación (AFF)', 'Arritmia (ARR)', 'Insuf. Cardíaca (CHF)', 'Normal (NSR)'],
            "Probabilidad %": st.session_state.probs
        })
        st.bar_chart(df_bar.set_index("Condición"), height=350)

        mejor = np.argmax(st.session_state.probs)
        st.success(f"**DIAGNÓSTICO PRELIMINAR: {df_bar['Condición'][mejor]}**")
        st.write(f"**Confianza: {st.session_state.probs[mejor]:.1f}%**")

# Iniciar el servidor (recepción + interfaz)
import uvicorn
if __name__ == "__main__":
    import threading
    threading.Thread(target=uvicorn.run, args=(api,), kwargs={"host":"0.0.0.0", "port":8000}).start()