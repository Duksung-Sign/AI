import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("videos/감사합니다02.mp4")

# CSV 저장 준비
csv_file = open('감사합니다02_좌표.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# 헤더: label + 각 좌표 이름
header = ['label']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']
csv_writer.writerow(header)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            row = ['감사합니다']
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            csv_writer.writerow(row)

    # 시각화 (선택)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
