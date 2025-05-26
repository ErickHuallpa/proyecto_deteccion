# Sistema de Detección de Sueño y Ebriedad usando Visión por Computadora

Este proyecto utiliza OpenCV y MediaPipe para detectar el estado de una persona (Despierto, Durmiendo o Posible Ebriedad) en tiempo real mediante análisis del parpadeo (EAR) y movimientos faciales. Además, emplea una red neuronal entrenada con datos simulados para clasificar el estado y dispara una alarma sonora cuando detecta sueño.

---

## Requisitos

- Python 3.8 o superior
- Webcam
- Archivo de audio `alarma.mp3` en la carpeta del proyecto

---

## Instalación de dependencias

```bash
pip install opencv-python mediapipe pygame numpy pandas scikit-learn tensorflow joblib
```

## Estructura del proyecto
bash
Copiar
Editar
proyecto_deteccion/
├── datos/
│   └── dataset.csv            # Dataset simulado (generado automáticamente)
├── generar_dataset.py         # Script para generar dataset simulado
├── modelo_entrenamiento.py   # Entrena la red neuronal con el dataset
├── modelo_entrenado.h5       # Modelo entrenado (se genera)
├── scaler.pkl                # Escalador de datos (se genera)
├── alarma.mp3                # Archivo de audio para la alarma
└── main_deteccion.py         # Script principal de detección en vivo

## Uso
1. Generar el dataset simulado

- Ejecuta el script para crear datos simulados en datos/dataset.csv:

```bash
python generar_dataset.py
```
2. Entrenar la red neuronal

- Ejecuta el script para entrenar el modelo con el dataset generado. Esto creará modelo_entrenado.h5 y scaler.pkl:

```bash
python modelo_entrenamiento.py
```
3. Ejecutar la detección en tiempo real

- Inicia la aplicación de detección en vivo que usa la webcam:

```bash
python main_deteccion.py
```
### Controles:
- ESC: Salir del programa
- ESPACIO: Apagar alarma manualmente

## Notas
Asegúrate de que alarma.mp3 se encuentre en la misma carpeta que main_deteccion.py.
La red neuronal usa 4 características: EAR izquierdo, EAR derecho, EAR promedio y movimiento facial para clasificar el estado.
Los datos simulados representan tres estados: Despierto (0), Durmiendo (1) y Posible Ebriedad (2).

## Autor
Erick Huallpa Vargas

