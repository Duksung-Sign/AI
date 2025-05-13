import cv2
import hand_tracking as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,     # 감지할 손 개수
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5

)
mp_draw = mp.solutions.drawing_utils

## cap = cv2.VideoCapture(0)  # 웹캠
cap=cv2.VideoCapture("videos/감사합니다01_거울.mp4")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Tracking", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
