from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

np.random.seed(1)
X = np.random.rand(40, 1)**2
y = (10-1./(X.ravel()+0.1))+np.random.randn(40)
X_test = np.linspace(-0.1,1.1,500).reshape(-1,1)

'''
GridSearchCV 객체를 만들면서 pipeline도 만들어서 쓸거고 파라미터 그리드고 써야함
파라미터 정보를 갖는 변수를 만듬, 아래 파라미터들은 규제가 있는 선형 회귀 모델 수업에서 자세히 설명할 예정 
params는 gridsearchcv함수의 매개변수이다.
'''
params = {"polynomialfeatures__degree": np.arange(21),
          "linearregression__fit_intercept": [True, False],
          }
#그리드 서치 객체 생성
grid = GridSearchCV(make_pipeline(StandardScaler(), PolynomialFeatures(), LinearRegression()), params, cv=7)
#이전 버전에서는 sklearn 라이브러리 linearregression모델에 normalize라는 매개변수가 있었으나 업데이트 되면서 사라짐.
#따라서 정규화를 하고자 한다면 StandardScaler()라는 preprocessing을 PolynomialFeatures과 같이 해야함.  
grid.fit(X, y)
grid.best_params_
optimal_model = grid.best_estimator_

plt.figure(figsize=(8,7))
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_pred = optimal_model.fit(X,y).predict(X_test)
plt.plot(X_test.ravel(), y_pred, "r-", linewidth=1, label="prediction line")
plt.axis(lim)
plt.legend(loc = "best")
plt.show()
