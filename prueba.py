import cv2
import mediapipe as mp
import numpy as np
import threading
import pygame


pygame.mixer.init()
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def reproducir_alarma():
    global alarma_sonando, alarma_desactivada
    if not alarma_sonando and not alarma_desactivada:
        alarma_sonando = True
        try:
            pygame.mixer.music.load('alarma.mp3')
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"Error al reproducir la alarma: {e}")

cap = cv2.VideoCapture(0)

contador_dormido = 0
UMBRAL_FRAMES_SUEÑO = 15
UMBRAL_EAR = 0.18
movimientos_erraticos = 0
UMBRAL_MOVIMIENTO = 15
UMBRAL_EBRIEDAD_FRAMES = 40

ultima_posicion_cara = None
alarma_sonando = False
alarma_desactivada = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    estado = "Desconocido"
    color = (255, 255, 255)

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
                if movimiento > UMBRAL_MOVIMIENTO:
                    movimientos_erraticos += 1
                else:
                    movimientos_erraticos = max(0, movimientos_erraticos - 1)

            ultima_posicion_cara = centro_cara

            if avg_ear < UMBRAL_EAR:
                contador_dormido += 1
            else:
                contador_dormido = 0
                alarma_desactivada = False
                if alarma_sonando:
                    pygame.mixer.music.stop()
                    alarma_sonando = False

            if contador_dormido >= UMBRAL_FRAMES_SUEÑO:
                estado = "DURMIENDO"
                color = (0, 0, 200)
                reproducir_alarma()
            elif avg_ear < 0.20 and movimientos_erraticos > UMBRAL_EBRIEDAD_FRAMES:
                estado = "POSIBLEMENTE EBRIO"
                color = (0, 0, 255)
                print("Alerta: Posible persona ebria detectada")
                if alarma_sonando:
                    pygame.mixer.music.stop()
                    alarma_sonando = False
            else:
                estado = "Despierto"
                color = (0, 255, 0)
                if alarma_sonando:
                    pygame.mixer.music.stop()
                    alarma_sonando = False

            cv2.putText(frame, f"Estado: {estado}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Deteccion de Suenio / Ebriedad", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        if alarma_sonando:
            pygame.mixer.music.stop()
            alarma_sonando = False
            alarma_desactivada = True
            print("Alarma apagada manualmente")

cap.release()
cv2.destroyAllWindows()