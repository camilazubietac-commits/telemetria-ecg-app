import streamlit as st
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ==================== ESTILO Y FONDO ====================
st.set_page_config(page_title="Vitals Link", layout="wide")

st.markdown("""
<style>
    .big-title {font-size: 80px !important; font-weight: bold; color: #006400; text-align: center; margin-bottom: 0px;}
    .sub-title {font-size: 28px !important; color: #000000; text-align: center; margin-top: 5px; font-weight: normal;}
    .hr-big {font-size: 70px !important; color: #006400; text-align: center; font-weight: bold;}
    .diag-title {font-size: 32px !important; color: #006400;}
    body {background: linear-gradient(135deg, #e8f5e9, #f1f8e8);}
</style>
""", unsafe_allow_html=True)

# Fondo sutil con ondas cardíacas (opcional, queda precioso)
st.markdown("""
<div style="position:fixed; top:0; left:0; width:100%; height:100%; opacity:0.03; pointer-events:none; 
     background-image: url('https://i.imgur.com/5fX9X8k.png'); background-size: cover; z-index:-1;"></div>
""", unsafe_allow_html=True)

# ==================== TÍTULO Y SUBTÍTULO ENORME ====================
# ==================== TÍTULO Y SUBTÍTULO QUE CAMBIAN DE COLOR SEGÚN DIAGNÓSTICO ====================

# Calculamos el color según la patología más probable
idx = np.argmax(st.session_state.data["probs"])
colores = ["#d32f2f", "#f57c00", "#c62828", "#2e7d32"]   # AFF, ARR, CHF, NSR
color_diagnostico = colores[idx]

# TÍTULO GRANDE que cambia de color
st.markdown(f'''
<h1 style="color:{color_diagnostico}; font-size:80px; text-align:center; margin-bottom:0; font-weight:bold;">
Vitals Link
</h1>
''', unsafe_allow_html=True)

# SUBTÍTULO que también cambia al mismo color
st.markdown(f'''
<h3 style="color:{color_diagnostico}; font-size:28px; text-align:center; margin-top:5px; font-weight:normal; opacity:0.9;">
El guardián digital de tu ritmo cardíaco, conectado y seguro.
</h3>
''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== CARGA DEL MODELO ====================
modelo = joblib.load("modelo_ecg_random_forest_final.pkl")
CLASES = ['Fibrilación Auricular', 'Arritmia', 'Insuficiencia Cardíaca', 'Ritmo Sinusal Normal']

# Estado inicial
if "data" not in st.session_state:
    st.session_state.data = {
        "I": [0]*600, "II": [0]*600, "III": [0]*600,
        "hr": 0, "features": {}, "probs": [25,25,25,25]
    }

# ==================== API FASTAPI ====================
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
# 3 gráficas en fila
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Derivación I")
    st.line_chart(st.session_state.data["I"], height=220, use_container_width=True)
with col2:
    st.subheader("Derivación II")
    st.line_chart(st.session_state.data["II"], height=220, use_container_width=True)
with col3:
    st.subheader("Derivación III (calculada)")
    st.line_chart(st.session_state.data["III"], height=220, use_container_width=True)

# Frecuencia cardíaca gigante
st.markdown(f'<p class="hr-big">{st.session_state.data["hr"]} <small style="font-size:50px;">bpm</small></p>', 
            unsafe_allow_html=True)

# Tabla de 12 features
st.subheader("12 Features Extraídos")
if st.session_state.data["features"]:
    feat_df = pd.DataFrame.from_dict(st.session_state.data["features"], orient='index', columns=['Valor'])
    st.dataframe(feat_df.style.format("{:.4f}"), use_container_width=True)
else:
    st.info("Esperando datos del ESP32...")

# Diagnóstico + justificación
st.markdown("<br><div class='diag-title'>Diagnóstico Automático</div>", unsafe_allow_html=True)
probs = st.session_state.data["probs"]
idx = np.argmax(probs)
patologia = CLASES[idx]
confianza = probs[idx]

colA, colB = st.columns([2, 3])
with colA:
    colores = ["#d32f2f", "#f57c00", "#388e3c", "#2e7d32"]
    st.markdown(f"<h2 style='color:{colores[idx]}; text-align:center;'>{patologia}<br>{confianza:.1f}%</h2>", 
                unsafe_allow_html=True)

with colB:
    just = f"""
    • Frecuencia cardíaca: {st.session_state.data['hr']} bpm  
    • Duración QRS: {st.session_state.data['features'].get('QRSseg',0):.3f} s → {'prolongado' if st.session_state.data['features'].get('QRSseg',0)>0.11 else 'normal'}  
    • Variabilidad RR (SDRR): {st.session_state.data['features'].get('SDRR',0):.3f} → {'alta' if st.session_state.data['features'].get('SDRR',0)>0.06 else 'normal'}  
    → Conclusión: el modelo detecta **{patologia.lower()}** con alta confianza.
    """
    st.write(just)

# ==================== SERVIDOR ====================
import uvicorn
if __name__ == "__main__":
    import threading
    threading.Thread(target=uvicorn.run, args=(api,), kwargs={"host":"0.0.0.0", "port":8000}).start()