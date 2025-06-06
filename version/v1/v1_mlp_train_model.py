import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# === 1. CSV 파일 병합 ===
def load_csv_data(csv_root):
    X_list, y_list = [], []

    for label in os.listdir(csv_root):
        label_path = os.path.join(csv_root, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if not file.endswith(".csv"):
                continue

            df = pd.read_csv(os.path.join(label_path, file))

            if df.shape[1] != 127:  # label + 126 좌표
                continue  # 잘못된 파일 생략

            X_list.append(df.iloc[:, 1:].values)  # 좌표만 (126차원)
            y_list.extend(df.iloc[:, 0].values)   # 라벨만

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

# === 2. 데이터 로딩 ===
X, y = load_csv_data("csv_data")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# === 3. 전처리: 스케일링 + 라벨 인코딩 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 4. 학습/검증 데이터 분리 ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# === 5. MLPClassifier 학습 ===
clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# === 6. 평가 ===
y_pred = clf.predict(X_test)
print("\n=== 분류 리포트 ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === 7. 모델 + 스케일러 + 라벨 저장 ===
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/sign_mlp_model.pkl")
joblib.dump(scaler, "models/sign_scaler.pkl")
joblib.dump(le, "models/sign_label_encoder.pkl")
print("\n✅ 모델 저장 완료!")
