import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import dump

# CSV 파일 로드
df = pd.read_csv('test.csv', header=None, names=['ax', 'ay', 'az', 'rx', 'ry', 'rz', 'label'])

# 데이터와 레이블 분리
X = df[['ax', 'ay', 'az', 'rx', 'ry', 'rz']]
y = df['label']

# 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 모델 리스트
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear')
}

# 모델 학습 및 교차 검증 평가
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=3)  # 3-겹 교차 검증 사용 (데이터가 매우 적으므로)
    results[name] = np.mean(scores)
    print(f"{name} Cross-Validation Accuracy: {np.mean(scores)}")

# 가장 성능이 좋은 모델 선택
best_model_name = max(results, key=results.get)
best_model_accuracy = results[best_model_name]
best_model = models[best_model_name]

print(f"Best Model: {best_model_name} with Cross-Validation Accuracy: {best_model_accuracy}")

# 모델 및 스케일러 저장
best_model.fit(X, y)  # 전체 데이터를 사용하여 최종 학습
dump(best_model, 'model.joblib')
dump(scaler, 'scaler.joblib')

# 보고서 작성
report = f"""
모델 성능 비교 보고서
======================

사용된 모델:
1. Logistic Regression
2. Random Forest
3. SVM

모델 성능 평가 (교차 검증 정확도):
- Logistic Regression: {results["Logistic Regression"]}
- Random Forest: {results["Random Forest"]}
- SVM: {results["SVM"]}

선정된 모델:
- {best_model_name}

선정 이유:
- {best_model_name} 모델이 {best_model_accuracy}의 가장 높은 교차 검증 정확도를 보였습니다.

결론:
- {best_model_name} 모델이 주어진 데이터셋에서 가장 우수한 성능을 보였으므로, 최종적으로 이 모델을 선택하였습니다.
"""

print(report)

with open("model_comparison_report.txt", "w") as f:
    f.write(report)
