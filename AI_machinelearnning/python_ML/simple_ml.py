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
train_score, val_score = validation_curve(make_pipeline(PolynomialFeatures(degree=2),LinearRegression()), X, y, param_name="polynomialfeatures__degree", param_range=degree, cv=7)

plt.figure(figsize=(8,5))
plt.plot(degree,np.median(train_score, axis=1),"b-", label = "train_score" )
plt.plot(degree,np.median(val_score, axis=1),"r-", label = "validation_score" )
plt.ylim(0,1)
plt.xlabel("degree")
plt.ylabel("score")
plt.legend(loc = "best")
plt.show()