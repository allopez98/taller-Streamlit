import streamlit as st
import subprocess
import sys

st.title("Modelo de Fatiga en Ciclismo")

st.write("Entrenar y evaluar modelos de Machine Learning")

if st.button("Entrenar modelos"):
    resultado = subprocess.run(
        [sys.executable, "train.py"], capture_output=True, text=True
    )
    if resultado.returncode == 0:
        st.success("Modelos entrenados correctamente")
        if resultado.stdout:
            st.text(resultado.stdout)
    else:
        st.error("Error al entrenar los modelos")
        st.text(resultado.stderr or resultado.stdout)

if st.button("Evaluar modelos"):
    resultado = subprocess.run(
        [sys.executable, "predict.py"], capture_output=True, text=True
    )
    if resultado.returncode == 0:
        st.text(resultado.stdout)
    else:
        st.error("Error al evaluar los modelos")
        st.text(resultado.stderr or resultado.stdout)