import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
from collections import deque

# === 모델 & 라벨 인코더 불러오기 ===
model = load_model("models/v2_sign_lstm_model.h5")
label_encoder = joblib.load("models/v2_sign_lstm_label_encoder.pkl")

# === MediaPipe 초기화 ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# === 시퀀스 설정 ===
SEQ_LENGTH = 30
sequence = deque(maxlen=SEQ_LENGTH)

# === 상대 좌표 변환 함수 ===
def get_relative_landmarks(landmarks):
    base = landmarks[0]
    return [(x - base[0], y - base[1]) for x, y in landmarks]

# === 손 & 얼굴 좌표 추출 함수 ===
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(image_rgb)
    results_face = face.process(image_rgb)

    keypoints = []

    # 손 좌표
    for hand_landmarks in (results_hand.multi_hand_landmarks or []):
        hand = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        keypoints += get_relative_landmarks(hand)
    while len(keypoints) < 42:  # 21점 x 2D
        keypoints += [(0.0, 0.0)]

    # 얼굴 (눈, 코, 입, 턱)
    face_points = []
    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0].landmark
        indices = [33, 263, 1, 61, 291, 199]  # 양눈, 코끝, 입 양끝, 턱
        face_points = [(face_landmarks[i].x, face_landmarks[i].y) for i in indices]
        face_points = get_relative_landmarks(face_points)
    while len(face_points) < 6:
        face_points += [(0.0, 0.0)]

    return np.array(keypoints + face_points).flatten()

# === 웹캠 실행 ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = extract_keypoints(frame)
    sequence.append(keypoints)

    # 시퀀스가 충분할 때 예측
    if len(sequence) == SEQ_LENGTH:
        input_seq = np.expand_dims(sequence, axis=0)  # (1, 30, feature)
        pred = model.predict(input_seq)[0]
        pred_label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        conf = np.max(pred)

        # 결과 표시
        cv2.putText(frame, f"{pred_label} ({conf:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-time Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
