#시각화된 최적모델(Visualized Optimal Model)
from sklearn.model_selection import validation_curve
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

np.random.seed(1)
X = np.random.rand(40, 1)**2
y = (10-1./(X.ravel()+0.1))+np.random.randn(40)

degree = np.arange(0, 21)
train_score, val_score = validation_curve(make_pipeline(PolynomialFeatures(degree=2),LinearRegression()), X, y, param_name="polynomialfeatures__degree", param_range=degree, cv=5)

plt.figure(figsize=(8,5))
plt.plot(degree,np.median(train_score, axis=1),"b-", label = "train_score" )
plt.plot(degree,np.median(val_score, axis=1),"r-", label = "validation_score" )
plt.ylim(0,1)
plt.xlabel("degree")
plt.ylabel("score")
plt.legend(loc = "best")
plt.show()
#위 코드에서 최적의 degree를 얻음
X_test = np.linspace(-0.1,1.1,500).reshape(-1,1)
plt.figure(figsize=(8,7))
plt.scatter(X.ravel(),y)
lim = plt.axis()
#얻은 최적의 degree를 넣어 파이프 라인을 만들고 바로 fit시킨다. 왜냐하면 이미 최적의 값을 얻었으니까.
ml_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(X,y)
y_pred = ml_model.predict(X_test)

plt.plot(X_test.ravel(), y_pred, "r-", linewidth=1, label = "prediction line")
plt.legend(loc = "best")
plt.axis(lim)
plt.show()