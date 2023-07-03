#1.데이터 준비
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

rs = np.random.RandomState(10)
x = 10*rs.rand(100)
y= 3*x+2*rs.rand(100)
'''
plt.scatter(x,y,s = 10) 
plt.show()
'''
regr1 = LinearRegression()
#하이퍼파라미터가 default인 선형휘귀 객체생성
regr2 = LinearRegression(fit_intercept=True)
#fit_intercept=True라는 하이퍼파라미터를 제공받아 선형휘기 객체 생성 

X = x.reshape(-1,1)
regr1.fit(X,y)
#모델을 데이터에 적합시킴, 인자로 X와 y가 들어옴

#새로운 입력값으로 학습한 모델을 토대로 예측
X_new = np.linspace(-1,11,100)
X_new = X_new.reshape(-1,1)
y_pred = regr1.predict(X_new)
'''
plt.plot(X_new, y_pred, c = "red")
plt.scatter(x,y,s = 10) 
plt.show()
'''
#오차값으로 모델 평가 
rmse = np.sqrt(mean_squared_error(y,y_pred))
print(rmse)