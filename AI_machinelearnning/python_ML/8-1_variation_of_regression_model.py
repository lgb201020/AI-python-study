import pandas as pd
import numpy as np
#선형 회귀 모델 class를 import함
from sklearn.linear_model import LinearRegression
#데이터를 분리하기 위한 train_test_split를 import함
from sklearn.model_selection import train_test_split
#회귀 모델 성능 평가지수 RMSE를 사용하기 위해 mean_squared_error를 import함
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

#linear regression instance 생성
model=LinearRegression(fit_intercept=True)

#교차 검증을 위해 데이터 분활(훈련용 데이터와 테스트용 데이터로 툭장행렬 및 대상 벡터 분리)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

#linear regression instance 학습을 통해 선형 회귀 모델 구축
model.fit(X_train, y_train)

#모델에 새로운 임의의 데이터 적용(입녁 변수는 반드시 1X11 matrix이어야 한다.)
newdata = np.random.random(11)
random_test = model.predict(newdata.reshape(1,11))

#앞서 분리해 놓았던 test 데이터에 대한 예측값 출력
y_pred = model.predict(X_test)

#RMSE를 이용해서 모델의 성능 평가(소수점 두자리 수까지 반올림)
model_accuracy = np.round(np.sqrt(mean_squared_error(y_pred, y_test)),2)
print(model_accuracy)
