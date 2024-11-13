import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# Configuraciones de entorno
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define la ruta base donde se encuentran los modelos
base_path = os.path.join(os.getcwd(), 'Models')

# Cargar los modelos
model_path_1 = os.path.join(base_path,'densetnet_121.tflite')
model = tf.lite.Interpreter(model_path_1)
model.allocate_tensors()

# Función para predecir la imagen y devolver la etiqueta y la precisión
def imagePrediction(image):
    images = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    images = cv2.resize(images, (150, 150))
    images = images.reshape(1, 150, 150, 3)
    prd_idx = model.predict(images)
    prd_idx = np.argmax(prd_idx, axis=1)[0]
    modelpre = model.predict(images)
    accuracy = modelpre[0][prd_idx]

    if prd_idx == 0:
        label = "Mancha Negra"
    elif prd_idx == 1:
        label = "Cancro"
    elif prd_idx == 2:
        label = "Enverdecimiento"
    elif prd_idx == 3:
        label = "Saludable"
    else:
        label = "Unknown"

    return label, accuracy

# Interfaz de usuario con Streamlit
st.title("Prediction of Neurodegenerative Diseases")

# Inicializar el estado de sesión si no existe
if 'results' not in st.session_state:
    st.session_state.results = []

# Subir nuevas imágenes
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    new_results = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        
        label, accuracy = imagePrediction(image)
        st.write(f'Model predicts that this is a {label} with an accuracy of {accuracy:.2f}')
        
        # Guardar resultados nuevos
        new_results.append({
            'Image': uploaded_file.name,
            'Label': label,
            'Accuracy': f'{accuracy:.2f}'
        })

    # Agregar solo los resultados nuevos al estado de sesión
    st.session_state.results.extend(new_results)


