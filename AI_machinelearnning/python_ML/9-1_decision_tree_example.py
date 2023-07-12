import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot
from IPython.core.display import Image
from IPython.display import display

#iris 데이터 준비 및 학습 데이터 만들기
iris = load_iris()
iris.keys()
iris.data[:5]
iris.feature_names

X = iris.data[:,2:]
iris.target[:5]
np.unique(iris.target, return_counts = True)
iris.target_names
y = iris.target


#DecisionTreeClassifier 모델 객체 생성 및 적합화
"""
DecisionTreeClassifier에서 설정할 수 있는 hyper parameter
DecisionTreeClassifier(class_weight = None, criterion = 'gini', max_depth = 2,
                       max_features = None, max_leaf_nodes = None,
                       min_impurity_decrease = 0.0, min_impurity_split = None,
                       min_samples_leaf = 1, min_samples_split = 2,
                       min_weight_fraction_leaf = 0.0, presort = False, random_state = None,
                       splitter = 'best')
"""
tree = DecisionTreeClassifier(max_depth = 10)
tree.fit(X,y)

#트리 출력하기
export_graphviz(tree, out_file = "iris.dot", feature_names = iris.feature_names[2:],
                class_names=iris.target_names, rounded = True, filled = True, impurity= True)

graph = pydot.graph_from_dot_file("iris.dot")[0]
iris_png = graph.create_png()
Image(iris_png)