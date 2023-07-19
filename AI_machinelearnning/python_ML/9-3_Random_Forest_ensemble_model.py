"""
* : 노랑 - 중요내용 강조
** : 그 와중에 더 중요한 내용
*** : 빨강 - 특징중 한계점이나 주의사항 강조
? : 초록 - 질문
"""
"""
ensemble model의 배경
***decision tree의 가장 큰 문제점은 고분산(overfitting)을 갖음. 즉 훈련 데이터에 대한 과적합(overfitting)이 심함 -> 이에 대한 해결 방법으로 ensemble model을 사용
ensemble model의 정의 : 앙상블은 성능이 좋지 않은 알고리즘들을 결합시켜 종합한 결과 기존의 단일 모델을 더 좋게 개선한 모델이다.
*앙상블의 대표적인 방식으로는 Bagging(Bootstrap aggregating)과 randomforest 방식이 있다.


bagging 방식
훈련 데이터에서 bootstrapping한 샘플에 대해 모든 변수를 선택해 다수의 decision tree를 구성 
-> 하나의 예측기와 편향성은 비슷하나 분산이 줄어듬
***bagging의 한계점: 선택된 변수가 모든 트리에서 대체로 비슷해져 모델간의 상관관계를 갖게 되면 모든 모델을 종합한다고 해도 분산 감소 효과가 기대만큼 크지 않을 수 있음 
*간단히 말하자면 원래의 single decision tree를 여러개 만들어 학습하고 각각의 예측기의 예측결과를 모아서 new sample에 대한 예측을 출력하는 방식이다.*


Random Forest 방식(최근 가장 많이 쓰임)
특징 기반 bagging 방법을 적용한 decision tree ensemble model이다.
일반 bagging은 전체 특징을 입력으로 받아 학습하는데 반해 random forest는 특징을 무작위로, 선택적으로 사용한다.
*개별 tree간의 상관관계 문제를 피하기 위해 bootstrapping 과정에서 훈련 데이터로부터 전체 p개의 변수(특징)중 무작위 m개의 변수(특징)만 선택한다.
*분류문제에 대한 모델의 경우: m = sqrt(p) --- p의 제곱근 반올림 값
*회귀문제에 대한 모델의 경우: m = p/3 
**randomfrest의 무작위성 주입은 tree의 다양화, 편향을 손해보는 대신에 분산을 낮춤

*기존의 단일 결정트리와 달리 모든 특징에 대한 중요도를 계산함-> 랜덤포레스트의 중요도는 각 의사결정 트리의 특징 중요도를 합한 후 트리의 수로 나눈것, 합이 1이 되도록 결과값을 정규화함
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.core.display import Image
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

iris = load_iris()
# iris 변수로 iris 데이터를 받음

random_forest = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# 랜덤 포레스트 객체 생성, 하이퍼 파라미터로 n_estimators(개별 트리 개수) = 500, n_jobs(사용할 프로세스 수) = -1(사용 가능한 모든 프로세스 사용)

random_forest = random_forest.fit(iris.data, iris.target)
# iris의 모든 데이터를 넣어줌 즉 배제된 특징 없이 모든 특징에 대한 데이터를 넣어줌

for feature, importance in zip(iris.feature_names, random_forest.feature_importances_):
    print("{}:{}".format(feature, importance))
# 계산된 random forest의 특징 중요도 값을 출력
