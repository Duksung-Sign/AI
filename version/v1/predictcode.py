import cv2
import mediapipe as mp
import joblib
import numpy as np

# 모델 불러오기
model = joblib.load("sign_knn_model_v1.pkl")

# Mediapipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# 테스트할 영상 경로
video_path = "videos/마음의약속.mp4"
cap = cv2.VideoCapture(video_path)

coords_list = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            coords = []
            for lm in hand.landmark:
                coords += [lm.x, lm.y, lm.z]
            if len(coords) == 63:
                coords_list.append(coords)

cap.release()

# 예측
if coords_list:
    avg_coords = np.mean(coords_list, axis=0).reshape(1, -1)

    # KNN 거리 확인
    distances, _ = model.kneighbors(avg_coords)
    avg_distance = np.mean(distances)
    print(f"📏 평균 거리: {avg_distance:.4f}")

    # 임계값 비교
    if avg_distance > 0.15:
        print("🤖 인식 결과: 알 수 없음 (낯선 수어)")
    else:
        pred = model.predict(avg_coords)
        print(f"🤖 인식 결과: {pred[0]}")
else:
    print("⚠️ 손 인식 실패 또는 좌표 없음")


#     pred = model.predict(avg_coords)
#     print(f"[🤖 예측 결과] 이 수어는 → {pred[0]}")
# else:
#     print("⚠️ 손 인식 실패 또는 좌표가 부족합니다.")
