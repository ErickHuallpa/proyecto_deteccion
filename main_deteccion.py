import cv2
import mediapipe as mp
import numpy as np
import pygame
from tensorflow.keras.models import load_model
import joblib
import os
import logging
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    pygame.mixer.init()
    audio_disponible = True
except pygame.error as e:
    print(f"Advertencia: No se pudo inicializar el sistema de audio ({e}). Las alarmas no funcionarán.")
    audio_disponible = False

modelo = load_model('modelo_entrenado.h5')
scaler = joblib.load('scaler.pkl')

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

alarma_sonando = False
alarma_desactivada = False
ultima_posicion_cara = None

EAR_UMBRAL = 0.22
FRAMES_UMBRAL = 2
contador_frames_cierre = 0

tiempo_alarma_inicio = None

mostrar_reduciendo_velocidad = False
tiempo_inicio_mensaje = None
TIEMPO_ESPERA_MENSAJE = 3

TIEMPO_MINIMO_ALARMA = 3

def reproducir_alarma():
    global alarma_sonando, alarma_desactivada, tiempo_alarma_inicio
    if not alarma_sonando and not alarma_desactivada and audio_disponible:
        alarma_sonando = True
        tiempo_alarma_inicio = time.time()
        try:
            pygame.mixer.music.load('alarma.mp3')
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"Error al reproducir la alarma: {e}")

def detener_alarma():
    global alarma_sonando, tiempo_alarma_inicio, mostrar_reduciendo_velocidad, tiempo_inicio_mensaje
    if alarma_sonando:
        pygame.mixer.music.stop()
        alarma_sonando = False
        tiempo_alarma_inicio = None
        mostrar_reduciendo_velocidad = False
        tiempo_inicio_mensaje = None

def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

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
    mirando_lados = False

    frame_infrarrojo = frame.copy()

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )
            mp_drawing.draw_landmarks(
                image=frame_infrarrojo,
                landmark_list=face_landmarks,
                connections=None,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
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

            if nose.x < 0.25 or nose.x > 0.75:
                mirando_lados = True
            else:
                mirando_lados = False

            if avg_ear < EAR_UMBRAL:
                contador_frames_cierre += 1
            else:
                contador_frames_cierre = 0

            entrada = np.array([[left_ear, right_ear, avg_ear, movimiento]])
            entrada_escalada = scaler.transform(entrada)
            prediccion = modelo.predict(entrada_escalada, verbose=0)
            estado_id = np.argmax(prediccion)

            if estado_id == 0:
                estado = "Despierto"
                color = (0, 255, 0)
                if alarma_sonando and contador_frames_cierre == 0:
                    detener_alarma()
            elif estado_id == 1 and contador_frames_cierre >= FRAMES_UMBRAL:
                estado = "DURMIENDO"
                color = (0, 0, 200)
                reproducir_alarma()

            if alarma_sonando and tiempo_alarma_inicio is not None:
                tiempo_alarma_actual = time.time() - tiempo_alarma_inicio
                if tiempo_alarma_actual >= TIEMPO_MINIMO_ALARMA:
                    if not mostrar_reduciendo_velocidad:
                        mostrar_reduciendo_velocidad = True
                        tiempo_inicio_mensaje = time.time()
                if mostrar_reduciendo_velocidad and tiempo_inicio_mensaje is not None:
                    if time.time() - tiempo_inicio_mensaje >= TIEMPO_ESPERA_MENSAJE:
                        mostrar_reduciendo_velocidad = False

            if mirando_lados:
                cv2.putText(frame, "Mirando a los lados", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)

            cv2.putText(frame, f"Estado: {estado}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Mov: {movimiento:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if mostrar_reduciendo_velocidad:
                cv2.putText(frame, "Reduciendo Velocidad...", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    gray = cv2.cvtColor(frame_infrarrojo, cv2.COLOR_BGR2GRAY)
    frame_ir_colormap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    cv2.imshow("Vista Normal", frame)
    cv2.imshow("Vista Infrarrojo", frame_ir_colormap)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        if alarma_sonando:
            detener_alarma()
            alarma_desactivada = True
            print("🔇 Alarma apagada manualmente")

cap.release()
cv2.destroyAllWindows()
