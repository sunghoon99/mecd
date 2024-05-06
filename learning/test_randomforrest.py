import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# CSV 파일 로드
df = pd.read_csv('test.csv', header=None, names=['ax', 'ay', 'az', 'rx', 'ry', 'rz', 'label'])

# 데이터와 레이블 분리
X = df[['ax', 'ay', 'az', 'rx', 'ry', 'rz']]
y = df['label']

# 데이터를 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2)
model.fit(X_train, y_train)

# 테스트 세트로 모델 평가
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# 새로운 데이터 예측
new_data = pd.DataFrame([[0,0,0,0,0,40]], columns=['ax', 'ay', 'az', 'rx', 'ry', 'rz'])
new_data_scaled = scaler.transform(new_data)  # 스케일링
prediction = model.predict(new_data_scaled)
print("Predicted label:", prediction[0])

# 모델 및 스케일러 저장
dump(model, 'model.joblib')
dump(scaler, 'scaler.joblib')
