from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

np.random.seed(1)
X = np.random.rand(40,1) ** 2
y = (10 - 1, / (X.ravel() + 0.1)) + np.random.randn(40)

%matplotlib inline
import matplotlib.pyplot as plt


 