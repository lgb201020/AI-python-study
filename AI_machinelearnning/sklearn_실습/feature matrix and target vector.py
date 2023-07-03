"""
numpy를 통해 X와 y 생성
import numpy as np
rs = np.random.RandomState(10)
'''
RandomState(10)에서 우리는 seed 값으로 10을 줌 
난수를 생성 할때 동일안 난수를 생성하기 위해서는 난수 값이 동일 해야 하기 때문에 seed값을 넣어 동일한 이산난수가 나오도록 한다.
'''
x= 10*rs.rand(5)
#rand함수에 5개의 랜덤 정보를 만들고 거기에 10을 곱해 변수 X에 저장
y= 2*x - 1*rs.rand(5)
#(2x-1)*random값에 해당하는 범위의 랜덤 값이 생성 -> 변수 y에 저장
print((x.shape, y.shape))
# x.shape, y.shape은 원소가 5개인 일차 배열을 원소로 갖는 2차 배열

X = x.reshape(-1,1)
#X를 특징행렬로, y를 대상 백터로 쓸려면 shape을 바꿔줘야한다. 
#-> 행,열에대한 정보를 넘겨주는데 reshape(-1,1)에서 열을 1로 고정, 행은 나머지 정보들로 구성
print(X.shape)
#열은 1개로 고정, row는 5개인 5차 column vector가 됨
#reshape을 통해 1차 배열을 2차 배열 구조로 바꿔줌
"""

"""
pandas를 통해 X와 y생성

import seaborn as sns
iris = sns.load_dataset("iris")
#print(iris.info())
#print(iris.head())
X = iris.drop("species", axis = 1)
print(X.shape)
print(X.values)
y = iris["species"]
print(y.shape)
"""

"""Bunch객체를 이용한 특징 행렬과 대상 벡터의 생성"""
from sklearn.datasets import load_iris

iris = load_iris()
print(type(iris))
print(iris.keys())
print(iris.feature_names)
print(iris.data[:5])
print(iris.data.shape)
print(iris.target)
print(iris.target.shape)
