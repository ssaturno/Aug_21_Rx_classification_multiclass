import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de imágenes",
    page_icon="🤖"
)

st.title("🤖 Clasificador de imágenes con IA")
st.write("Sube una imagen y el modelo intentará reconocerla.")

# Evita notación científica
np.set_printoptions(suppress=True)


# Cargar modelo una sola vez
@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5", compile=False)


model = cargar_modelo()

# Cargar las clases
with open("labels.txt", "r", encoding="utf-8") as archivo:
    class_names = archivo.readlines()


# Subir imagen
archivo_subido = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)


if archivo_subido is not None:

    # Abrir imagen
    image = Image.open(archivo_subido).convert("RGB")

    # Mostrar imagen
    st.image(image, caption="Imagen seleccionada", width=300)

    # Ajustar tamaño
    size = (224, 224)
    image = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    # Convertir a numpy
    image_array = np.asarray(image)

    # Normalizar
    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    # Crear arreglo de entrada
    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image_array

    # Realizar predicción
    prediction = model.predict(data)

    # Obtener clase con mayor probabilidad
    index = np.argmax(prediction)

    class_name = class_names[index].strip()

    # Quitar el número que pone Teachable Machine
    class_name = class_name.split(" ", 1)[-1]

    confidence_score = prediction[0][index]

    # Mostrar resultado
    st.subheader("Resultado")

    st.success(
        f"Predicción: {class_name}"
    )

    st.write(
        f"Confianza: {confidence_score * 100:.2f}%"
    )
