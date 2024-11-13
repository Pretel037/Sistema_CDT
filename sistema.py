import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

# Información de enfermedades
disease_info = {
    "Mancha Negra": {
        "Nombre": "Mancha Negra",
        "Precisión": "95%",
        "Agente Causal": "Hongos",
        "Síntomas": "Manchas oscuras en hojas y frutos",
        "Recomendación": "Usar fungicidas y controlar la humedad"
    },
    "Cancro": {
        "Nombre": "Cancro",
        "Precisión": "92%",
        "Agente Causal": "Bacterias",
        "Síntomas": "Lesiones en hojas y ramas",
        "Recomendación": "Aplicar cobre y podar las áreas afectadas"
    },
    "Enverdecimiento": {
        "Nombre": "Enverdecimiento",
        "Precisión": "89%",
        "Agente Causal": "Bacterias transmitidas por insectos",
        "Síntomas": "Hojas amarillentas y frutos deformados",
        "Recomendación": "Control de insectos y plantas enfermas"
    },
    "Saludable": {
        "Nombre": "Saludable",
        "Precisión": "100%",
        "Agente Causal": "N/A",
        "Síntomas": "Sin síntomas",
        "Recomendación": "No se requiere tratamiento"
    }
}

# Cargar modelos con rutas relativas
model_paths = {
    "DenseNet121": "models/densetnet_121.keras",
}
models = {name: tf.keras.models.load_model(path) for name, path in model_paths.items()}

# Función para predecir usando el modelo seleccionado
def image_prediction(image, model):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (150, 150))
    image = image.reshape(1, 150, 150, 3)
    pred = model.predict(image)
    pred_class = np.argmax(pred, axis=1)[0]
    accuracy = pred[0][pred_class]

    labels = ["Mancha Negra", "Cancro", "Enverdecimiento", "Saludable"]
    return labels[pred_class], accuracy

# Interfaz de Streamlit
st.title("Sistema de Citrus CDT")

# Entrada de cámara
image = st.camera_input("Captura una imagen para analizar")

# Procesar la imagen y mostrar los resultados si se captura una imagen
if image:
    image_file = Image.open(image)

    # Seleccionar modelo y predecir
    selected_model_name = "DenseNet121"
    selected_model = models[selected_model_name]
    result, accuracy = image_prediction(image_file, selected_model)
    accuracy_text = f"{accuracy * 100:.2f}%"  # Convierte el valor de precisión en porcentaje

    # Mostrar imagen y tabla de información una al lado de la otra
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_file, caption=f"Predicción: {result} (Precisión: {accuracy_text})", use_column_width=True)

    with col2:
        st.header("Información de la Enfermedad")
        st.table({
            "Categoría": ["Nombre", "Precisión", "Agente Causal", "Síntomas", "Recomendación"],
            "Descripción": [
                disease_info[result]["Nombre"],
                accuracy_text,  # Usa la precisión calculada en lugar de la predefinida
                disease_info[result]["Agente Causal"],
                disease_info[result]["Síntomas"],
                disease_info[result]["Recomendación"]
            ]
        })

    # Mensaje de predicción y precisión
    st.write(f"El modelo predice que esto es {result} con una precisión de {accuracy_text}")
