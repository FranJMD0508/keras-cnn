# Perros vs. Gatos con Keras

Práctica de **Computación Emergente** (UJAP · Ing. en Computación · Prof. María García).

Clasificador binario de imágenes (gato / perro) entrenado desde cero con una CNN en Keras, sobre el dataset Kaggle Cats vs Dogs.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranJMD0508/keras-cnn/blob/main/perros_vs_gatos_keras.ipynb)

## Contenido

| Archivo | Descripción |
|---|---|
| `perros_vs_gatos_keras.ipynb` | Notebook ejecutado en Google Colab (GPU T4) |
| `.gitignore` | Excluye el dataset (786 MB) y los checkpoints `.keras` |

## Flujo del notebook

1. Descarga del dataset (~25.000 imágenes) y filtrado de imágenes corruptas.
2. `image_dataset_from_directory` con imágenes de 180×180 y 20 % para validación.
3. Data augmentation: volteo horizontal y rotación aleatoria.
4. Normalización de píxeles de [0, 255] a [0, 1] con `Rescaling`.
5. CNN tipo Xception reducida: `Conv2D` / `SeparableConv2D` + `BatchNormalization` + ReLU + `MaxPooling2D`, con conexiones residuales, `GlobalAveragePooling2D`, `Dropout(0.25)` y una neurona de salida.
6. Entrenamiento con Adam (3e-4) y `BinaryCrossentropy` durante **10 épocas** (el original usa 25).
7. Inferencia con `model.predict()` sobre una imagen nueva.

## Cómo ejecutarlo

Abrir con el botón de Colab, activar **Entorno de ejecución → Cambiar tipo de entorno → GPU T4** y luego **Ejecutar todo**.

## Créditos

Basado en el ejemplo oficial [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/) de François Chollet ([keras-team/keras-io](https://github.com/keras-team/keras-io), licencia Apache 2.0).
