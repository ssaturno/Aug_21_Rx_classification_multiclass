# Clasificador de Radiografías con Inteligencia Artificial

Este proyecto implementa un sistema de **clasificación de imágenes de radiografías de tórax utilizando Inteligencia Artificial y Deep Learning**.

El modelo fue entrenado utilizando **Google Teachable Machine** y posteriormente exportado en formato **Keras (`.h5`)** para integrarlo en una aplicación desarrollada con **Python y Streamlit**.

El objetivo del proyecto es demostrar de forma práctica cómo un modelo de clasificación de imágenes puede ser entrenado, exportado e integrado en una aplicación para realizar predicciones sobre nuevas imágenes.

> **Nota:** Este proyecto tiene fines educativos y demostrativos. No debe utilizarse como herramienta de diagnóstico médico.

## Clases del modelo

El modelo fue entrenado para reconocer cuatro categorías:

* **Covid-19**
* **Sano**
* **Neumonía Viral**
* **Neumonía Bacteriana**

Estas clases se encuentran almacenadas en el archivo `labels.txt`.

## Entrenamiento del modelo

Para el entrenamiento se utilizó **Google Teachable Machine**, una herramienta que permite crear modelos de Machine Learning y Deep Learning mediante una interfaz gráfica.

El proceso general realizado fue:

1. Crear un proyecto de clasificación de imágenes en Teachable Machine.
2. Definir las cuatro clases.
3. Cargar las imágenes correspondientes a cada categoría.
4. Entrenar el modelo.
5. Evaluar su comportamiento dentro de Teachable Machine.
6. Exportar el modelo en formato **TensorFlow/Keras**.
7. Obtener los archivos `keras_model.h5` y `labels.txt`.
8. Integrar el modelo en una aplicación desarrollada con Streamlit.

El archivo `keras_model.h5` contiene el modelo neuronal entrenado, incluyendo su arquitectura y los pesos aprendidos durante el entrenamiento.

## Aplicación con Streamlit

Para facilitar el uso del modelo se desarrolló una interfaz web utilizando **Streamlit**.

La aplicación carga el archivo `keras_model.h5` utilizando Keras y también carga las clases desde `labels.txt`.

El usuario puede subir una imagen en formato:

* JPG
* JPEG
* PNG

Una vez cargada la imagen, la aplicación realiza automáticamente el procesamiento necesario antes de enviarla al modelo.

## Preprocesamiento de las imágenes

Las imágenes deben tener el mismo formato de entrada utilizado por el modelo.

Por esta razón, la aplicación:

1. Convierte la imagen a formato **RGB**.
2. Ajusta su tamaño a **224 × 224 píxeles**.
3. Convierte la imagen en un arreglo de NumPy.
4. Convierte los valores a `float32`.
5. Normaliza los píxeles utilizando:

```python
(image_array.astype(np.float32) / 127.5) - 1
```

6. Construye un arreglo de entrada con dimensiones:

```text
(1, 224, 224, 3)
```

Este procedimiento puede observarse directamente en la implementación de la aplicación.

## Predicción

Después del preprocesamiento, la imagen es enviada al modelo:

```python
prediction = model.predict(data)
```

El modelo genera una probabilidad para cada una de las cuatro clases.

Posteriormente se utiliza `np.argmax()` para identificar la clase con la probabilidad más alta y se obtiene también su nivel de confianza.

Por ejemplo, el resultado podría mostrarse como:

```text
Resultado

Neumonia Viral

Confianza del modelo
92.45%
```

La versión final de la aplicación también permite visualizar las probabilidades obtenidas para **todas las clases**, no únicamente la predicción principal.

## Estructura del proyecto

```text
proyecto/
│
├── app.py
├── keras_model.h5
├── labels.txt
├── requirements.txt
└── README.md
```

### `app.py`

Contiene la aplicación desarrollada con Streamlit. Se encarga de cargar el modelo, recibir la imagen, realizar el preprocesamiento, ejecutar la predicción y mostrar los resultados.

La interfaz presenta la imagen seleccionada y el resultado de la clasificación en dos columnas.

### `keras_model.h5`

Modelo de Deep Learning entrenado y exportado desde **Teachable Machine**.

### `labels.txt`

Contiene las clases utilizadas por el modelo:

```text
0 Covid-19
1 Sano
2 Neumonia Viral
3 Neumonia Bacteriana
```

### `requirements.txt`

Contiene las dependencias necesarias para ejecutar el proyecto:

```text
streamlit
tensorflow==2.15.0
keras==2.15.0
pillow
numpy<2
```

## Instalación

Primero se deben instalar las dependencias:

```bash
pip install -r requirements.txt
```

Luego se puede iniciar la aplicación utilizando:

```bash
streamlit run app.py
```

Streamlit abrirá una interfaz web desde la cual se podrá cargar una radiografía y realizar la clasificación.

## Flujo del proyecto

```text
Radiografía
     ↓
Carga de imagen con Streamlit
     ↓
Conversión a RGB
     ↓
Redimensionamiento a 224 × 224
     ↓
Conversión a arreglo NumPy
     ↓
Normalización de los píxeles
     ↓
Modelo Keras entrenado en Teachable Machine
     ↓
Predicción de las 4 clases
     ↓
Selección de la mayor probabilidad
     ↓
Resultado + porcentaje de confianza
```

## Tecnologías utilizadas

* **Google Teachable Machine** — entrenamiento del modelo.
* **TensorFlow / Keras** — carga y ejecución del modelo de Deep Learning.
* **Python** — desarrollo de la aplicación.
* **Streamlit** — interfaz web.
* **NumPy** — procesamiento de arreglos y resultados.
* **Pillow (PIL)** — procesamiento y redimensionamiento de imágenes.

## Objetivo educativo

Este proyecto permite comprender de manera práctica el flujo completo de una solución básica de **clasificación de imágenes con Deep Learning**:

**Datos → entrenamiento → modelo → exportación → preprocesamiento → inferencia → aplicación**

Teachable Machine simplifica la etapa de entrenamiento, mientras que Python, TensorFlow/Keras y Streamlit permiten utilizar posteriormente el modelo dentro de una aplicación funcional.

De esta manera, el ejercicio sirve como introducción a conceptos como **clasificación multiclase, redes neuronales para imágenes, preprocesamiento, inferencia y despliegue de modelos de Inteligencia Artificial**.

## Consideraciones

El porcentaje mostrado por la aplicación corresponde a la **confianza de la predicción del modelo**, y no debe interpretarse como una probabilidad clínica de que una persona tenga determinada enfermedad.

El desempeño del sistema depende directamente de factores como la calidad y representatividad de las imágenes utilizadas para entrenar el modelo, el balance entre las clases y las condiciones bajo las cuales se capturaron las imágenes.

Por esta razón, el proyecto debe considerarse una **demostración académica del funcionamiento de un clasificador de imágenes basado en Deep Learning**, no un sistema médico validado.
