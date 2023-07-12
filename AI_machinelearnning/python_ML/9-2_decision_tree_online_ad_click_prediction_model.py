
"""
transformation gz file to csv file
import pandas as pd
path_gzip_file = 
print("작동 준비 중")
gzip_file_data_frame = pd.read_csv(path_gzip_file, compression='gzip', header=0, sep=",", quotechar='"')
print("작동 완료")
print(gzip_file_data_frame.head())

# CSV 파일로 저장
csv_file_path = 'output.csv'
gzip_file_data_frame.to_csv(csv_file_path, index=False)

print("CSV 파일이 성공적으로 저장되었습니다.")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import pydot
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


#온라인 광고 클릭을 위한 훈련 데이터 준비
train_df = pd.read_csv("C:\\Users\\기백이 노트북\\Desktop\\pythonstudy\\AI_machinelearnning\\python_ML\\train.gz", compression='gzip', header=0, sep=",", quotechar='"', nrows=100000)
unused_columns, label_column = ["id", "hour", "device_id", "device_ip"], "click"
train_df = train_df.drop(unused_columns, axis=1)

X_dict_train = list(train_df.drop(label_column, axis=1).T.to_dict().values()) #target vector인 label_column(click)을 뺀 나머지 데이터를 학습 데이터로 사용

y_train = train_df[label_column] #target vector인 label_column(click)을 y변수로 선언



#온라인 광고 클릭을 위한 테스트 데이터 준비
"""
X라는 특징 행렬을 만들기 전 가공 단계
1. train_df에서 label column을 drop해서 데이터 준비
2. T는 matrix transpose 기능 메서드, to_dict() 메서드는 df를 dictionary 형태로 변환 -> values만 반환
3. 2에서의 반환받은 values를 list type으로 변환 -> list로 변환(벡터로 쓰기 위해) 
"""
#test_df에서 header를 0으로 선언하여 첫번째 line을 사용함을 명시적으로 표현
test_df = pd.read_csv("C:\\Users\\기백이 노트북\\Desktop\\pythonstudy\\AI_machinelearnning\\python_ML\\train.gz", compression='gzip', header=0, sep=",", quotechar='"', skiprows=(1,100000), nrows=100000)
test_df = test_df.drop(unused_columns, axis=1)
X_dict_test = list(test_df.drop(label_column, axis=1).T.to_dict().values())
y_test = test_df[label_column]



#DictVectorizer 객체 생성 후 앞에서 가공한 X_dict_train과 X_dict_test를 벡터로 변환 후 X_train, X_test로 각각 선언
"""
원 핫 인코딩 벡터 변환
sklearn-tree model은 입력 변수로 사용하는 모든 특징들을 수치로 입력받는다. 즉 범주형 데이터는 수치형으로 뱐환이 필요
따라서 앞서 dictionary로 변환한 입력변수들을 list로 변환한것도 인코딩 벡터 변환을 모든 value에 일괄적으로 하기 위해서이다.
"""
vectorizer = DictVectorizer(sparse=True)
#sparse=True는 벡터화 한 특징들을 희소행렬로 반환 여부에 대한 parameter이다.
#gridsearch를 통해 최적의 parameter와 model을 찾을 때 메모리를 많이 쓰는데 특징 행렬을 그냥 행렬로 쓰면 불필요하게 0이 많아지고 그만큼 메모리 사용이 증가하기 때문에 
#희소행렬도 불필요 0을 없애서 메모리 사용량을 최소화 
X_train = vectorizer.fit_transform(X_dict_train)
X_test = vectorizer.transform(X_dict_test)
#벡터로 변환하면 19개의 column(차원)이 4952차원으로 증가함



#의사 결정 트리(CART 알고리즘을 이용) 학습 모델을 선언하고 gridsearchCV로 최적의 모델 탐색

parameters = {'max_depth': [3, 10, None]} #gridsearchCV에 사용할 hyperparams를 미리 선언

decision_tree = DecisionTreeClassifier(criterion="gini",min_samples_split=30) #기본 decision tree model을 생성

#gridsearch객체 선언 후 학습 ml 모델 입력, 하이퍼 파라미터 입력, scoring 방식 입력(roc_auc는 분류평가지표 중 하나로 ROC곡선 아래의 면적을 의미)
grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring="roc_auc")

grid_search.fit(X_train, y_train) #선언한 gridsearch객체에 학습 데이터 적합

grid_search.best_params_ #최적의 hyperparams를 출력

decision_tree_best = grid_search.best_estimator_ #grid_search 객체로 찾은 최적의 모델 인스턴스 선언




#의사 결정 트리 모델의 파일 출력
#decision tree graph 생성을 위한 함수 호출로 dot파일 생성
export_graphviz(decision_tree_best, out_file="ctr_decision_tree.dot", feature_names=vectorizer.feature_names_, 
                class_names=["0","1"], rounded=True, filled=True, impurity= True)



#dot 파일을 기반으로 그래프 생성 및 파일 저장
graph = pydot.graph_from_dot_file("ctr_decision_tree.dot")
graph.write_png("ctr_decision_tree.png")



#모델의 성능 측정
#예측값 출력
y_pred = decision_tree_best.predict(X_test)
np.unique(y_pred, return_counts=True)

#성능 측정
accuracy_score(y_test, y_pred) #정확도

confusion_matrix(y_test, y_pred) #confusion matrix 생성

y_pred_proba = decision_tree_best.predict_proba(X_test)[:,1] #클릭할 확률과 클릭 안할 확률 중 클릭할 확률만 출력

fpr, tpr = roc_curve(y_test, y_pred_proba) #roc curve X,y값 선언

auc = roc_auc_score(y_test, y_pred_proba) #roc의 auc 계산



#ROC curve 그래프 출력
plt.plot(fpr, tpr, "r-", label = "Decision tree Classifier") #ROC curve 그래프 그리기

plt.plot([0, 1], [0, 1], "b--", label = "random guess") #random guess 그래프 그리기

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC{0:.2f}".format(auc))
plt.legend(loc="lower right")

plt.show()

