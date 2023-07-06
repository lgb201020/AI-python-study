import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Lasso linear regression model class를 import함
from sklearn.linear_model import Lasso
#데이터를 분리하기 위한 train_test_split를 import함
from sklearn.model_selection import train_test_split
#회귀 모델 성능 평가지수 RMSE를 사용하기 위해 mean_squared_error를 import함.
from sklearn.metrics import mean_squared_error


#데이터 로드
red_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";", header=0)
red_wine["type"] = "red"

white_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";", header=0)
white_wine["type"] = "white"

#필요 데이터 분류, X(특징 행렬-matrix(nparray의 2차 배열 구조))과 y(대상 벡터-vector(nparray의 1차 배열 구조))로 데이터 구성
wine = pd.concat([red_wine, white_wine], axis=0)
wine.columns = wine.columns.str.replace(" ", "_")

X = wine.drop(["type","quality"], axis=1) 
y =wine.quality

#교차 검증을 위해 데이터 분활(훈련용 데이터와 테스트용 데이터로 툭장행렬 및 대상 벡터 분리)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

#모델 hyperparameter 설정
model_name = "Lasso_model"
alpha = 0.05

#matlab 시각화  fig, ax 설정
fig = plt.figure(figsize=(6,3))
ax =fig.add_subplot(111)

#Lasso 모델 인스턴스 생성 및 학습, 테스트 값 출력
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

#Lasso 모델 성능 평가지수 RMSE 출력 및 학습 모델 계수 출력, 시각화
RMSE = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
coef = pd.Series(data = lasso.coef_, index= X_train.columns).sort_values()
ax.bar(coef.index, coef.values)
ax.set_xticklabels(coef.index, rotation=90) #rotation=90을 써서 X라벨을 세로로 씀
ax.set_title("{0} : alpha = {1}, RMSE = {2}".format(model_name, alpha, RMSE));
plt.show()
