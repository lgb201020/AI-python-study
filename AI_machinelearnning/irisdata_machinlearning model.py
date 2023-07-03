import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 25)


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(np.mean(y_test == y_pred))
print(knn.score(X_test,y_test))
print(accuracy_score(y_test,y_pred))