import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === 1. CSV에서 X, y 불러오기 ===
csv_files = glob.glob("감사합니다*landmarks_with_handedness.csv")
X_list, y_list = [], []

for path in csv_files:
    df = pd.read_csv(path, encoding='utf-8-sig')
    label = df['label'].iloc[0].split('_')[0]  # '감사합니다'
    coords = df.drop(['label', 'hand'], axis=1).values
    X_list.append(coords)
    y_list.append(np.full(len(coords), label))

X = np.vstack(X_list)
y = np.concatenate(y_list)

print("X shape:", X.shape)
print("y classes:", np.unique(y))

# === 2. KNN 학습 함수 정의 ===
def run_knn_classifier(X, y, n_neighbors=3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("\n=== 분류 리포트 ===")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()
    plt.show()
    joblib.dump(model, "sign_knn_model_v1.pkl")
    print("✅ 모델 저장 완료: sign_knn_model_v1.pkl")

# === 3. 실제 함수 호출 ===
run_knn_classifier(X, y, n_neighbors=3)
# 모델 학습 후 저장
