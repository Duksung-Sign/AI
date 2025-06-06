import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image

# === 한글 출력 함수 ===
def draw_korean_text(img, text, position=(30, 50), font_size=40, color=(0, 200, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "C:/Windows/Fonts/H2GTRE.TTF"  # HY견고딕 경로
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# === 손 랜드마크를 상대 좌표로 변환 ===
def get_relative_landmarks(hand_landmarks):
    base = hand_landmarks[0]
    relative = []
    for lm in hand_landmarks:
        relative.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])
    return relative

# === 모델, 스케일러, 라벨 인코더 로딩 ===
clf = joblib.load("models/sign_mlp_model.pkl")
scaler = joblib.load("models/sign_scaler.pkl")
label_encoder = joblib.load("models/sign_label_encoder.pkl")

# === MediaPipe 초기화 ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# === 웹캠 시작 ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    label_text = ""

    if result.multi_hand_landmarks:
        features = []

        # 두 손 좌표 모두 추출
        for hand_landmarks in result.multi_hand_landmarks:
            coords = get_relative_landmarks(hand_landmarks.landmark)
            if len(coords) == 63:
                features.append(coords)

        # 양손 다 있을 때만 예측 진행
        if len(features) == 2:
            full_feature = features[0] + features[1]  # 총 126차원
            X = np.array(full_feature).reshape(1, -1)
            X_scaled = scaler.transform(X)

            probs = clf.predict_proba(X_scaled)[0]
            confidence = np.max(probs)

            if confidence < 0.7:
                label_text = "모르겠음"
            else:
                pred = clf.predict(X_scaled)
                label_text = label_encoder.inverse_transform(pred)[0]

            # 한글 출력
            frame = draw_korean_text(frame, f"👉 {label_text}", position=(30, 50))

        # 랜드마크 시각화
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
