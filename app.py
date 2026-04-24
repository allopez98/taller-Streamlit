import streamlit as st
import subprocess

st.title("Modelo de Fatiga en Ciclismo")

st.write("Entrenar y evaluar modelos de Machine Learning")

if st.button("Entrenar modelos"):
    subprocess.run(["python", "train.py"])
    st.success("Modelos entrenados correctamente")

if st.button("Evaluar modelos"):
    resultado = subprocess.run(
        ["python", "predict.py"], capture_output=True, text=True
    )
    st.text(resultado.stdout)