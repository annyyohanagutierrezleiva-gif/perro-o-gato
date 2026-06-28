import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Perro o Gato IA", layout="centered")
st.title("Clasificador Perro o Gato - Clase IA - 2026 - Anny Gutierrez")
st.write("Suba una imagen para clasificar con el modelo MobileNetV2 pre-entrenado")

IMG_SIZE = (224, 224)
CLASES = ["Gato", "Perro"]

@st.cache_resource
def cargar_modelo():
    return tf.saved_model.load("modelo_savedmodel")

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return tf.expand_dims(arr, axis=0)

def predecir(img):
    tensor = preparar_imagen(img)
    preds = modelo(tensor, training=False).numpy()[0]
    indice = np.argmax(preds)
    return CLASES[indice], float(preds[indice]) * 100, preds

modelo = cargar_modelo()

archivo = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen analizada", use_container_width=True)
    resultado, confianza, preds = predecir(imagen)
    st.subheader("Resultado")
    st.success(f"Predicción: {resultado} ({confianza:.2f}%)")
    st.write("Probabilidades:")
    for i, clase in enumerate(CLASES):
        st.write(f"{clase}: {float(preds[i])*100:.2f}%")
else:
    st.info("Cargue una imagen para iniciar la clasificación.")
