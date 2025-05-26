# Proyecto Detección de Somnolencia y Ebriedad

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-red)

---

## Descripción

Este proyecto implementa un sistema de **detección en tiempo real** de somnolencia y ebriedad utilizando visión por computadora y aprendizaje automático. 

Utiliza la librería [MediaPipe](https://mediapipe.dev/) para la detección de landmarks faciales y calcula el **Eye Aspect Ratio (EAR)** y movimiento facial para determinar el estado del usuario:

- **Despierto**
- **Durmiendo**
- **Posiblemente ebrio**

Para mejorar la precisión, se entrena una red neuronal que clasifica el estado basado en características extraídas en tiempo real.

---

## Características principales

- Captura de video en tiempo real con OpenCV.
- Detección de landmarks faciales con MediaPipe Face Mesh.
- Cálculo de métricas faciales: Eye Aspect Ratio (EAR) y movimientos faciales.
- Recolección manual de datos para etiquetado y entrenamiento.
- Entrenamiento de red neuronal con TensorFlow/Keras.
- Alarma sonora activada cuando se detecta somnolencia.
- Clasificación automática del estado usando modelo entrenado.
- Código modular y fácil de entender.

---

## Estructura del proyecto

```plaintext
proyecto_deteccion/
│
├── datos/
│   └── dataset.csv             # Datos recolectados para entrenamiento
├── alarma.mp3                  # Sonido de alarma para somnolencia
├── main_deteccion.py           # Código principal para detección en tiempo real
├── modelo_entrenamiento.py     # Script para entrenar el modelo
├── modelo_entrenado.h5         # Modelo entrenado (guardado)
├── recolectar_datos.py         # Script para recolectar datos y generar CSV
├── scaler.pkl                  # Archivo con scaler para normalizar datos
├── README.md                   # Este archivo
└── requirements.txt            # Dependencias del proyecto
