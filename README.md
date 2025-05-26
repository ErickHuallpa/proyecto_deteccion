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



## Estructura del proyecto


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
Instalación
Se recomienda usar un entorno virtual de Python.

bash
Copiar
Editar
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
Dependencias principales
opencv-python

mediapipe

numpy

pygame

tensorflow

scikit-learn

pandas

joblib

## Uso
1. Recolectar datos
Ejecuta el script para recolectar datos con etiquetas manuales.

bash
Copiar
Editar
python recolectar_datos.py
Presiona D para marcar "Despierto"

Presiona S para marcar "Durmiendo"

Presiona E para marcar "Posiblemente ebrio"

Presiona ESC para salir

Los datos se guardan en datos/dataset.csv.

2. Entrenar el modelo
Ejecuta el script para entrenar la red neuronal:

bash
Copiar
Editar
python modelo_entrenamiento.py
El modelo entrenado se guardará en modelo_entrenado.h5 y el scaler en scaler.pkl.

3. Ejecutar detección en tiempo real
Corre el script principal:

bash
Copiar
Editar
python main_deteccion.py
Se abrirá la cámara y se mostrará el estado detectado en tiempo real con una alarma sonora en caso de somnolencia.

Personalización
Puedes ajustar los umbrales y parámetros en el código.

Modificar el modelo para agregar más características o estados.

Cambiar el archivo de audio alarma.mp3 por otro sonido.

Licencia
Este proyecto está bajo la licencia MIT. Puedes usarlo y modificarlo libremente.

Contacto
Erick Huallpa - GitHub
Correo: erickhuallpa@example.com

¡Gracias por usar el proyecto! Si tienes dudas o sugerencias, abre un issue o PR en GitHub.
