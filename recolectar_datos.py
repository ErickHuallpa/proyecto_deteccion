import cv2
import mediapipe as mp
import numpy as np
import csv

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

cap = cv2.VideoCapture(0)
csv_file = open('datos/dataset.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['left_ear', 'right_ear', 'avg_ear', 'movimiento', 'label'])

ultima_posicion_cara = None

print("Presiona D (despierto), S (dormido), E (ebrio), ESC para salir")

left_ear = right_ear = avg_ear = movimiento = 0  # Variables por defecto para cuando no haya rostro

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Dibujar rostro
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
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

    cv2.imshow("Recolección de datos", frame)
    key = cv2.waitKey(1) & 0xFF
    label = None
    if key == ord('d'):
        label = 0
    elif key == ord('s'):
        label = 1
    elif key == ord('e'):
        label = 2
    elif key == 27:
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()
        break

    # Guardar si hay etiqueta y landmarks detectados
    if label is not None and result.multi_face_landmarks:
        csv_writer.writerow([left_ear, right_ear, avg_ear, movimiento, label])
        print(f"Guardado: EAR={avg_ear:.2f}, Movimiento={movimiento:.2f}, Etiqueta={label}")
