import os
import cv2
import numpy as np


# === 증강 함수들 ===
def flip(frame):
    return cv2.flip(frame, 1)


def brighten(frame, value=30):
    return cv2.convertScaleAbs(frame, alpha=1.0, beta=value)


def darken(frame, value=30):
    return cv2.convertScaleAbs(frame, alpha=1.0, beta=-value)


def rotate(frame, angle):
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(frame, matrix, (w, h))


def shift(frame, x=10, y=10):
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def blur(frame, k=5):
    return cv2.GaussianBlur(frame, (k, k), 0)


# === 증강 목록 정의 ===
AUGMENTATIONS = {
    "flip": flip,
    "bright1": lambda f: brighten(f, 30),
    "bright2": lambda f: brighten(f, 60),
    "dark": lambda f: darken(f, 30),
    "rot5": lambda f: rotate(f, 5),
    "rot-5": lambda f: rotate(f, -5),
    "shift": shift,
    "blur": blur,
}


# === 영상 하나 증강 저장 ===
def apply_augmentations(video_path, save_dir, aug_dict):
    filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    # 영상 정보 불러오기
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 모든 프레임 저장
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # 원본 복사 저장
    out_origin = cv2.VideoWriter(os.path.join(save_dir, f"{filename}.mp4"), fourcc, fps, (w, h))
    for f in frames:
        out_origin.write(f)
    out_origin.release()

    # 증강 저장
    for key, func in aug_dict.items():
        out_path = os.path.join(save_dir, f"{filename}_{key}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames:
            aug_frame = func(f)
            out.write(aug_frame)
        out.release()
        print(f"✅ 저장됨: {out_path}")


# === 폴더 내 모든 영상 증강 ===
def process_all_videos(data_root="data", save_root="data_aug"):
    for label in os.listdir(data_root):
        label_path = os.path.join(data_root, label)
        if not os.path.isdir(label_path):
            continue

        output_label_path = os.path.join(save_root, label)
        os.makedirs(output_label_path, exist_ok=True)

        for file in os.listdir(label_path):
            if not file.endswith(".mp4"):
                continue

            input_path = os.path.join(label_path, file)
            apply_augmentations(input_path, output_label_path, AUGMENTATIONS)


if __name__ == "__main__":
    process_all_videos("data", "data_aug")
