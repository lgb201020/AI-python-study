import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
#*일반적으로 logistic regression은 이진 분류, 대규모 데이터를 다루는데 유리하다.

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

#훈련 데이터 불러오기
train_df = pd.read_csv("train.csv", nrows = 100000)
unused_columns, label_column = ["id", "hour", "device_id", "device_ip"], "click"
train_df = train_df.drop(unused_columns, axis=1)
X_dict_train = list(train_df.drop(label_column, axis=1).T.to_dict().values())
y_train = train_df[label_column]

#테스트 데이터 불러오기
test_df = pd.read_csv("train.csv", skiprows=(1, 100000), nrows=100000)
test_df = test_df.drop(unused_columns, axis=1)
X_dict_test = list(test_df.drop(label_column, axis=1).T.to_dict().values())
y_test = test_df[label_column]

#불러온 데이터 벡터화로 특징을 수치화하기
vectorizer = DictVectorizer(sparse=True)
#** DictVectorizer(sparse=True)는 sparse=True로 설정하는 것은 옵션이 아닌 필!수!이다.
X_train = vectorizer.fit_transform(X_dict_train)
X_test = vectorizer.transform(X_dict_test)

#logistic regression model object 생성
clf = LogisticRegression(max_iter = 1000, solver = "saga")
#*solver는 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'가 있는데 이중에서 penalty가 l1,l2모두 가능한게 saga이다.
#*solver가 한 값에 수렴하기 위해 입력받는 데이터의 최대값을 설정해줘야한다. default는 100이다.
#*LogisticRegression()은 C와 penalty를 parameters로 갖고 있다. C는 L1,L2 규제 강도를 지정하는 매개변수로 C가 0에 가까울 수록 규제 강도가 강해진다.
#*penalty는 L1,L2 중 어떠한 규제 알고리즘을 사용할지 지정하는 매개변수이다.
#*L1는 계수값이 항상 0보다 크거나 같다, L2는 계수값이 항상 0보다 크다. 이 둘의 차이는 L1만 계수값이 0이 될 수 있다는 것이다.
#*L1, L2에서 패널티는 overfitting을 방지하고자 사용한다.
clf.fit(X_train, y_train)

#grid search로 최적의 모델 찾기
parameters = {"C": [0.001, 0.01, 0.1, 1, 10], "penalty": ["l1","l2"]}
#하이퍼 파라미터로 C에 규제 강도를 입력하고 penalty에 L1규제를 적용할지 L2규제를 적용할지 정하기 위해 리스트 형테로 둘다 넣음
grid_search = GridSearchCV(clf, parameters, n_jobs=1, cv=3, scoring= "roc_auc")
#/* 로컬로 실행시 시스템 내부 설정으로 인해 n_jobs = -1이 불가하고 colab으로 실행시 -1이 가능하다.
grid_search.fit(X_train, y_train)

#최적의 하이퍼 파라미터 출력
clf_best = grid_search.best_estimator_
y_pred = clf_best.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_pred_proba = clf_best.predict_proba(X_test)[:,1]
print("정확도:{}, 예측확률:{}".format(accuracy,y_pred_proba))
#반응변수는 positive probability를 값으로 갖으며 possitive probability는 예측값의 두변째 열에 있다.

#모델 성능 시각화
fpr, tpr, unused_element = roc_curve(y_test, y_pred_proba)
auc = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, "r-", label="Logistic Regression")
plt.plot([0, 1], [0, 1], "b--", label = "random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC={}".format(auc))
plt.show()

#*Qt 라이브러리가 지금 5.15.6 버전이라 5.15.8 버전이 계속 충돌해서 시각화 모듈이 작동을 안한다. 해결하자! -> 걍 아나콘다 밀고 다시 설치하면 됨

