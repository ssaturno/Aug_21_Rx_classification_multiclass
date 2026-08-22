import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# ---------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------

st.set_page_config(
    page_title="Clasificador de imágenes con IA",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------------------------------
# ESTILOS
# ---------------------------------------------------

st.markdown("""
<style>

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

.titulo {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitulo {
    text-align: center;
    font-size: 18px;
    color: #888;
    margin-bottom: 30px;
}

.resultado {
    padding: 20px;
    border-radius: 12px;
    background-color: rgba(120,120,120,0.1);
    text-align: center;
    margin-top: 20px;
}

.resultado h2 {
    margin-bottom: 5px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# TÍTULO
# ---------------------------------------------------

st.markdown(
    '<div class="titulo">🤖 Clasificador de imágenes con IA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitulo">'
    'Sube una imagen y el modelo identificará a qué categoría pertenece.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------
# CARGAR MODELO
# ---------------------------------------------------

np.set_printoptions(suppress=True)


@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5", compile=False)


model = cargar_modelo()


# ---------------------------------------------------
# CARGAR ETIQUETAS
# ---------------------------------------------------

with open("labels.txt", "r", encoding="utf-8") as archivo:
    class_names = archivo.readlines()


# ---------------------------------------------------
# SUBIR IMAGEN
# ---------------------------------------------------

archivo_subido = st.file_uploader(
    "📷 Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)


if archivo_subido is not None:

    image = Image.open(archivo_subido).convert("RGB")

    # Dos columnas
    columna1, columna2 = st.columns(2)

    with columna1:

        st.subheader("Imagen")

        st.image(
            image,
            use_container_width=True
        )


    # ---------------------------------------------------
    # PREPARAR IMAGEN
    # ---------------------------------------------------

    size = (224, 224)

    imagen_procesada = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(imagen_procesada)

    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image_array


    # ---------------------------------------------------
    # PREDICCIÓN
    # ---------------------------------------------------

    prediction = model.predict(data)

    index = np.argmax(prediction)

    class_name = class_names[index].strip()

    # Elimina el número de la etiqueta
    class_name = class_name.split(" ", 1)[-1]

    confidence_score = prediction[0][index]

    porcentaje = confidence_score * 100


    # ---------------------------------------------------
    # RESULTADO
    # ---------------------------------------------------

    with columna2:

        st.subheader("Resultado")

        st.markdown(
            f"""
            <div class="resultado">
                <h2>{class_name}</h2>
                <p>Confianza del modelo</p>
                <h3>{porcentaje:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(confidence_score))


    # ---------------------------------------------------
    # TODAS LAS PREDICCIONES
    # ---------------------------------------------------

    with st.expander("Ver probabilidades de todas las clases"):

        for i, probabilidad in enumerate(prediction[0]):

            nombre = class_names[i].strip()
            nombre = nombre.split(" ", 1)[-1]

            st.write(
                f"{nombre}: {probabilidad * 100:.2f}%"
            )
