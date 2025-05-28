import cv2
import mediapipe as mp
import numpy as np
import pygame
from tensorflow.keras.models import load_model
import joblib
import os
import logging
import tensorflow as tf

# Configurar logging para suprimir mensajes no deseados
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Inicializar pygame con manejo de errores
try:
    pygame.mixer.init()
    audio_disponible = True
except pygame.error as e:
    print(f"Advertencia: No se pudo inicializar el sistema de audio ({e}). Las alarmas no funcionarán.")
    audio_disponible = False

# Cargar modelo y scaler
modelo = load_model('modelo_entrenado.h5')
scaler = joblib.load('scaler.pkl')

# Configurar Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Definiciones de constantes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UMBRAL_MOVIMIENTO = 15

# Variables de estado
alarma_sonando = False
alarma_desactivada = False
ultima_posicion_cara = None

def reproducir_alarma():
    global alarma_sonando, alarma_desactivada
    if not alarma_sonando and not alarma_desactivada and audio_disponible:
        alarma_sonando = True
        try:
            pygame.mixer.music.load('alarma.mp3')
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"Error al reproducir la alarma: {e}")

def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

# Capturar video
cap = cv2.VideoCapture('ejemplo.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    estado = "Desconocido"
    color = (255, 255, 255)
    movimiento = 0.0

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )

            left_eye = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in LEFT_EYE])
            right_eye = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in RIGHT_EYE])
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            nose = face_landmarks.landmark[1]
            centro_cara = np.array([nose.x * w, nose.y * h])
            if ultima_posicion_cara is not None:
                movimiento = np.linalg.norm(centro_cara - ultima_posicion_cara)
            ultima_posicion_cara = centro_cara

            entrada = np.array([[left_ear, right_ear, avg_ear, movimiento]])
            entrada_escalada = scaler.transform(entrada)
            
            # Suprimir salida de predict
            prediccion = modelo.predict(entrada_escalada, verbose=0)
            estado_id = np.argmax(prediccion)

            if estado_id == 0:
                estado = "Despierto"
                color = (0, 255, 0)
                if alarma_sonando:
                    pygame.mixer.music.stop()
                    alarma_sonando = False
            elif estado_id == 1:
                estado = "DURMIENDO"
                color = (0, 0, 200)
                reproducir_alarma()
            elif estado_id == 2:
                estado = "POSIBLEMENTE EBRIO"
                color = (0, 0, 255)
                print("⚠️ Alerta: Posible persona ebria detectada")
                if alarma_sonando:
                    pygame.mixer.music.stop()
                    alarma_sonando = False

            cv2.putText(frame, f"Estado: {estado}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Mov: {movimiento:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Deteccion de Suenio / Ebriedad", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        if alarma_sonando:
            pygame.mixer.music.stop()
            alarma_sonando = False
            alarma_desactivada = True
            print("🔇 Alarma apagada manualmente")

cap.release()
cv2.destroyAllWindows()