import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

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

# Define la ruta base donde se encuentran los modelos
base_path = os.path.join(os.getcwd(), 'models')

# Diccionario para almacenar modelos cargados
models = {}

# Función para cargar un modelo Keras o TFLite
def load_model(model_name, model_path):
    if model_path.endswith('.tflite'):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        models[model_name] = interpreter
    else:
        models[model_name] = tf.keras.models.load_model(model_path)

# Cargar modelos (puedes agregar más modelos aquí)
load_model("DenseNet121", os.path.join(base_path, 'densetnet_121.tflite'))

# Función para predecir usando el modelo seleccionado
def image_prediction(image, model):
    # Preprocesar imagen
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (265, 265))
    image = image.reshape(1, 265, 265, 3).astype(np.float32)

    # Predicción dependiendo del tipo de modelo
    if isinstance(model, tf.keras.Model):  # Para modelos Keras
        pred = model.predict(image)
    else:  # Para modelos TFLite
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], image)
        model.invoke()
        pred = model.get_tensor(output_details[0]['index'])

    # Obtener clase y precisión
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
