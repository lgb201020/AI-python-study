import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from IPython.core.display import Image
from IPython.display import display
import multiprocessing
import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

num_cores = multiprocessing.cpu_count()
print("Number of CPU cores:", num_cores)
# 멀티 코어로 머신러닝 학습 활성화


# 온라인 광고 클릭을 위한 훈련 데이터 준비
train_df = pd.read_csv("train.csv", nrows=100000)
unused_columns, label_column = ["id", "hour", "device_id", "device_ip"], "click"
train_df = train_df.drop(unused_columns, axis=1)

X_dict_train = list(train_df.drop(label_column, axis=1).T.to_dict().values())
y_train = train_df[label_column]
# target vector인 label_column(click)을 y변수로 선언
# 온라인 광고 클릭을 위한 테스트 데이터 준비
"""
X라는 특징 행렬을 만들기 전 가공 단계
1. train_df에서 label column을 drop해서 데이터 준비
2. T는 matrix transpose 기능 메서드, to_dict() 메서드는 df를 dictionary 형태로 변환 -> values만 반환
3. 2에서의 반환받은 values를 list type으로 변환 -> list로 변환(벡터로 쓰기 위해) 
"""
# test_df에서 header를 0으로 선언하여 첫번째 line을 사용함을 명시적으로 표현
print("온라인 광고 클릭을 위한 훈련 데이터 준비완료")

test_df = pd.read_csv("train.csv", skiprows=(1, 100000), nrows=100000)
test_df = test_df.drop(unused_columns, axis=1)
X_dict_test = list(test_df.drop(label_column, axis=1).T.to_dict().values())
y_test = test_df[label_column]
# DictVectorizer 객체 생성 후 앞에서 가공한 X_dict_train과 X_dict_test를 벡터로 변환 후 X_train, X_test로 각각 선언
"""
원 핫 인코딩 벡터 변환
sklearn-tree model은 입력 변수로 사용하는 모든 특징들을 수치로 입력받는다. 즉 범주형 데이터는 수치형으로 뱐환이 필요
따라서 앞서 dictionary로 변환한 입력변수들을 list로 변환한것도 인코딩 벡터 변환을 모든 value에 일괄적으로 하기 위해서이다.
"""


vectorizer = DictVectorizer(sparse=True)
# sparse=True는 벡터화 한 특징들을 희소행렬로 반환 여부에 대한 parameter이다.
# gridsearch를 통해 최적의 parameter와 model을 찾을 때 메모리를 많이 쓰는데 특징 행렬을 그냥 행렬로 쓰면 불필요하게 0이 많아지고 그만큼 메모리 사용이 증가하기 때문에
# 희소행렬도 불필요 0을 없애서 메모리 사용량을 최소화

X_train = vectorizer.fit_transform(X_dict_train)
X_test = vectorizer.transform(X_dict_test)
# 벡터로 변환하면 19개의 column(차원)이 4952차원으로 증가함
print("온라인 광고 클릭을 위한 훈련 데이터 원 핫 인코딩 벡터 변환 완료")

random_forest = RandomForestClassifier(
    n_estimators=100, criterion="gini", min_samples_split=30, n_jobs=1
)
# 랜덤 포레스트 객체 생성
# 하이퍼 파라미터로 n_estimators(개별 트리 개수) = 100, 불순도 = "gini", 분류되는 sample의 최소값 =30, n_jobs(사용할 프로세스 수) = -1(사용 가능한 모든 프로세스 사용)
print("random_forest 객체 생성 완료")

parameters = {"max_depth": [3, 10, None]}
# gridsearchCV에 사용할 hyperparams를 미리 선언

grid_search = GridSearchCV(random_forest, parameters, n_jobs=1, cv=3, scoring="roc_auc")
# gridsearchCV 객체 생성
# 모델: random_forest, 파라미터: parameters, n_jobs: 가능한 모든 프로세스 사용, cv: 3겹 교차검증, scoring: ROC곡선의 AUC(곡선하면적)
print("grid_search model generated")

grid_search.fit(X_train, y_train)
# gridsearchCV 객체에 train 데이터 fitting
print("train data fitting complete")

bp = grid_search.best_params_
print(bp)
# 가장 최적화된 모델의 파라미터 출력

random_forest_best = grid_search.best_estimator_
# 가장 최적화된 모델 객체 생성
# ***scikit-learn 1.2.2 버전에서는 best_params_, best_estimator_를 사용하지 않는다. 1.3.0버전에서 사용한다.***


# 모델의 성능 측정
# 예측값 출력
y_pred = random_forest_best.predict(X_test)
np.unique(y_pred, return_counts=True)
print("prediction complete")

# 성능 측정
accuracy_score(y_test, y_pred)
print("accuracy_score : {}".format(accuracy_score(y_test, y_pred)))
# random_forest_best 모델의 정확도 출력

confusion_matrix(y_test, y_pred)
print("confusion_matrix:{}".format(confusion_matrix(y_test, y_pred)))
# random_forest_best 모델의 confusion matrix 생성

y_pred_proba = random_forest_best.predict_proba(X_test)[:, 1]
print("클릭할 확률과 클릭 안할 확률 중 클릭할 확률:{}".format(y_pred_proba))
# 클릭할 확률과 클릭 안할 확률 중 클릭할 확률만 출력

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# roc curve X,y값 선언, roc_curve함수는 fpr, tpr, threshold(임계값)을 반환하기 때문에 "_"변수로 임계값을 받아준다.

auc = roc_auc_score(y_test, y_pred_proba)
# roc의 auc 계산
print("ROC curve의 AUC 계산 완료")


# ROC curve 그래프 출력
plt.plot(fpr, tpr, "r-", label="Decision tree Classifier")
# ROC curve 그래프 그리기
print
plt.plot([0, 1], [0, 1], "b--", label="random guess")  # random guess 그래프 그리기

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC{0:.2f}".format(auc))
plt.legend(loc="lower right")

plt.show()
