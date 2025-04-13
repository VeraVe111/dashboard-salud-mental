
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción de Salud Mental", layout="centered")

st.title("Dashboard Predictivo: Salud Mental y Estilo de Vida")
st.markdown("Este modelo predice si una persona tiene probabilidad de presentar una condición de salud mental, según su estilo de vida.")

model = joblib.load("modelo_salud_mental.pkl")

age = st.slider("Edad", 18, 65)
sleep = st.slider("Horas de sueño", 0.0, 12.0, 7.0)
screen_time = st.slider("Horas frente a pantalla", 0.0, 12.0, 4.0)
work_hours = st.slider("Horas de trabajo/semana", 0, 80, 40)
social_score = st.slider("Puntaje de interacción social (0-10)", 0.0, 10.0, 5.0)
happiness = st.slider("Puntaje de felicidad (0-10)", 0.0, 10.0, 6.0)

data = pd.DataFrame([{
    "Age": age,
    "Sleep Hours": sleep,
    "Screen Time per Day (Hours)": screen_time,
    "Work Hours per Week": work_hours,
    "Social Interaction Score": social_score,
    "Happiness Score": happiness
}])

if st.button("Predecir"):
    prediction = model.predict(data)
    resultado = "Posible condición de salud mental." if prediction[0] == 1 else "Sin indicios de condición mental."
    st.success(resultado)
