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
import torch
import sys
from datetime import datetime

# Rutas y configuraciones
sys.path.append('./yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Inicializar audio
try:
    pygame.mixer.init()
    audio_disponible = True
except pygame.error as e:
    print(f"Advertencia: No se pudo inicializar audio ({e})")
    audio_disponible = False

# Cargar modelos
modelo = load_model('modelo_entrenado.h5')
scaler = joblib.load('scaler.pkl')
device = select_device('')
yolo_model = DetectMultiBackend('yolov5s.pt', device=device)
yolo_model.eval()
STRIDE = yolo_model.stride
CLASSES = yolo_model.names

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
MOUTH = [13, 14]  # Labios superior e inferior

# Variables de estado
alarma_sonando = False
alarma_desactivada = False
ultima_posicion_cara = None
EAR_UMBRAL = 0.22
FRAMES_UMBRAL = 2
contador_frames_cierre = 0
TIEMPO_MINIMO_ALARMA = 3
TIEMPO_ESPERA_MENSAJE = 3
tiempo_alarma_inicio = None
mostrar_reduciendo_velocidad = False
tiempo_inicio_mensaje = None

# Funciones auxiliares
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

def detectar_objetos(frame, focal=700, ancho_real_objeto=0.5):
    img = letterbox(frame, 640, stride=STRIDE)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = yolo_model(img, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None)[0]
    detecciones = []

    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in pred:
            cls_name = CLASSES[int(cls)]
            x1, y1, x2, y2 = map(int, xyxy)
            ancho_px = x2 - x1
            if ancho_px > 0:
                distancia = (ancho_real_objeto * focal) / ancho_px
                detecciones.append((cls_name, (x1, y1, x2, y2), distancia))
    return detecciones

# Capturas
cap1 = cv2.VideoCapture("ejemplo.mp4")
cap2 = cv2.VideoCapture("video.mp4")
os.makedirs("capturas", exist_ok=True)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1:
        break
    if not ret2:
        frame2 = np.zeros_like(frame1)

    h, w = frame1.shape[:2]
    rgb_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    estado = "Desconocido"
    color = (255, 255, 255)
    movimiento = 0.0
    mirando_lados = False

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Solo dibujar puntos clave
            for idx in LEFT_EYE + RIGHT_EYE + [NOSE_TIP] + MOUTH:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame1, (x, y), 2, (0, 255, 255), -1)

            left_eye = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in LEFT_EYE])
            right_eye = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in RIGHT_EYE])
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            nose = face_landmarks.landmark[NOSE_TIP]
            centro_cara = np.array([nose.x * w, nose.y * h])
            if ultima_posicion_cara is not None:
                movimiento = np.linalg.norm(centro_cara - ultima_posicion_cara)
            ultima_posicion_cara = centro_cara

            mirando_lados = nose.x < 0.25 or nose.x > 0.75

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
                if time.time() - tiempo_alarma_inicio >= TIEMPO_MINIMO_ALARMA:
                    if not mostrar_reduciendo_velocidad:
                        mostrar_reduciendo_velocidad = True
                        tiempo_inicio_mensaje = time.time()
                if mostrar_reduciendo_velocidad and tiempo_inicio_mensaje is not None:
                    if time.time() - tiempo_inicio_mensaje >= TIEMPO_ESPERA_MENSAJE:
                        mostrar_reduciendo_velocidad = False

            cv2.putText(frame1, f"Estado: {estado}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame1, f"EAR: {avg_ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame1, f"Mov: {movimiento:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if mirando_lados:
                cv2.putText(frame1, "Mirando a los lados", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)
            if mostrar_reduciendo_velocidad:
                cv2.putText(frame1, "Reduciendo Velocidad...", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    objetos = detectar_objetos(frame2)
    alerta_mostrada = False
    for cls_name, (x1, y1, x2, y2), dist in objetos:
        cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame2, f"{cls_name}: {dist:.2f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if cls_name in ['person', 'car'] and dist <= 5.0:
            alerta_mostrada = True
            cv2.putText(frame2, f"{cls_name.upper()} CERCA!!!", (w//4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            filename = f"capturas/alerta_{cls_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame2)

    cv2.imshow("Fatiga y Distracción", frame1)
    cv2.imshow("Detección Objetos", frame2)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        if alarma_sonando:
            detener_alarma()
            alarma_desactivada = True
            print("🔇 Alarma apagada manualmente")

cap1.release()
cap2.release()
cv2.destroyAllWindows()
