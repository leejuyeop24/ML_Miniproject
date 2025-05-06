# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
DATA_FILE = "../data/ObesityDataSet_raw_and_data_sinthetic.csv"
DF = pd.read_csv(DATA_FILE, usecols=[4, 5, 6, 7, 8, 9, 11], engine='python')

# 피처 및 타겟 설정
feature = DF[['FAVC', 'FCVC']]  # 예측 변수: 패스트푸드 섭취(FAVC), 채소 소비 빈도(FCVC)
target = DF['BMI']  # 목표 변수: BMI

# 데이터 분할 (훈련 70%, 테스트 30%)
train_X, test_X, train_y, test_y = train_test_split(feature, target, test_size=0.3, random_state=42)

# 선형 회귀 모델 학습
LinearModel = LinearRegression()
LinearModel.fit(train_X, train_y)

# 예측 수행
pre_y_train = LinearModel.predict(train_X)
pre_y_test = LinearModel.predict(test_X)

# 모델 성능 평가
train_r2 = r2_score(train_y, pre_y_train)
test_r2 = r2_score(test_y, pre_y_test)

train_rmse = np.sqrt(mean_squared_error(train_y, pre_y_train))
test_rmse = np.sqrt(mean_squared_error(test_y, pre_y_test))

# 모델 결과 출력
print(f"📊 모델 적합도 (R² Score):")
print(f"  - 학습 데이터: {train_r2:.4f}")
print(f"  - 테스트 데이터: {test_r2:.4f}")

print(f"\n📉 RMSE (Root Mean Squared Error):")
print(f"  - 학습 데이터: {train_rmse:.4f}")
print(f"  - 테스트 데이터: {test_rmse:.4f}")

# 모델 회귀 계수 출력
print(f"\n📈 회귀 계수 (coefficient): {LinearModel.coef_}")
print(f"📉 절편 (intercept): {LinearModel.intercept_}")

# 시각화: 실제 BMI vs 예측 BMI
plt.figure(figsize=(10, 5))
plt.scatter(train_y, pre_y_train, alpha=0.5, label='Train Data', color='blue')
plt.scatter(test_y, pre_y_test, alpha=0.5, label='Test Data', color='red')
plt.plot([min(target), max(target)], [min(target), max(target)], linestyle='--', color='black', label='Ideal Fit')
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("BMI Prediction: Actual vs. Predicted")
plt.legend()
plt.show()

