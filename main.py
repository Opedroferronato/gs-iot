import cv2
import mediapipe as mp
import time
import math



TEMPO_OLHO_FECHADO = 2.0  # segundos para detectar fadiga
EAR_THRESHOLD = 0.20      # sensibilidade do olho
MAR_THRESHOLD = 0.60      # sensibilidade do bocejo



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



def distancia(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def calcular_EAR(landmarks, olho):

    vertical1 = distancia(landmarks[olho[1]], landmarks[olho[5]])
    vertical2 = distancia(landmarks[olho[2]], landmarks[olho[4]])
    horizontal = distancia(landmarks[olho[0]], landmarks[olho[3]])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def calcular_MAR(landmarks, boca):

    vertical = distancia(landmarks[boca[0]], landmarks[boca[1]])
    horizontal = distancia(landmarks[boca[2]], landmarks[boca[3]])

    mar = vertical / horizontal
    return mar


LEFT_EYE = [33, 160, 158, 133, 153, 144]


RIGHT_EYE = [362, 385, 387, 263, 373, 380]


MOUTH = [13, 14, 78, 308]


cap = cv2.VideoCapture(0)

tempo_inicio_fechado = None
status = "ATENTO"

while True:
    sucesso, frame = cap.read()

    if not sucesso:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultados = face_mesh.process(rgb)

    altura, largura, _ = frame.shape

    if resultados.multi_face_landmarks:

        for face_landmarks in resultados.multi_face_landmarks:

            landmarks = face_landmarks.landmark



            ear_esquerdo = calcular_EAR(landmarks, LEFT_EYE)
            ear_direito = calcular_EAR(landmarks, RIGHT_EYE)

            ear = (ear_esquerdo + ear_direito) / 2



            mar = calcular_MAR(landmarks, MOUTH)



            if ear < EAR_THRESHOLD:

                if tempo_inicio_fechado is None:
                    tempo_inicio_fechado = time.time()

                tempo_fechado = time.time() - tempo_inicio_fechado

                # Evita contar piscadas rápidas
                if tempo_fechado >= TEMPO_OLHO_FECHADO:
                    status = "ALERTA DE SONO"

            else:
                tempo_inicio_fechado = None
                status = "ATENTO"



            if mar > MAR_THRESHOLD:
                status = "BOCEJO DETECTADO"


            for ponto in LEFT_EYE + RIGHT_EYE + MOUTH:

                x = int(landmarks[ponto].x * largura)
                y = int(landmarks[ponto].y * altura)

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)



            cv2.putText(
                frame,
                f"STATUS: {status}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255) if status != "ATENTO" else (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    cv2.imshow("Sensor de Fadiga - GS", frame)

    tecla = cv2.waitKey(1)

    if tecla == 27:
        break

cap.release()
cv2.destroyAllWindows()