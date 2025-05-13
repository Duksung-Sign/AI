import cv2
import mediapipe as mp
import csv

video_path = "videos/감사합니다09.mp4"
csv_path = "감사합니다09_landmarks_with_handedness.csv"
label = "감사합니다09"
show_video=True

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.1
)

cap = cv2.VideoCapture(video_path)

with open(csv_path, "w", newline='',encoding='utf-8-sig') as f:
    writer = csv.writer(f)

    # 헤더
    header = ['label', 'hand']
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']
    writer.writerow(header)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # 화면 기준 왼손/오른손 → 실제 손 기준으로 반전
                detected_label = result.multi_handedness[i].classification[0].label
                actual_hand = 'Right' if detected_label == 'Left' else 'Left'

                row = [label, actual_hand]
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                writer.writerow(row)

                # ==== 시각화 코드 ====
                if show_video:
                    mp_draw.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if show_video:
                    cv2.imshow("Hand Tracking", image)
                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        break

cap.release()
cv2.destroyAllWindows()
print(f"[✅ 저장 완료] → {csv_path}")