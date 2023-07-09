
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
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#온라인 광고 클릭을 위한 훈련 데이터 준비
train_df = pd.read_csv('C:\\Users\\기백이 노트북\\Desktop\\pythonstudy\\AI_machinelearnning\\python_ML\\train.gz', compression='gzip', header=0, sep=",", quotechar='"', nrows=100000)
unused_columns, label_column = ["id", "hour", "device_id", "device_ip"], "click"
train_df = train_df.drop(unused_columns, axis=1)
#target vector인 label_column(click)을 뺀 나머지 데이터를 학습 데이터로 사용
X_dict_train = list(train_df.drop(label_column, axis=1).T.to_dict().values())
#target vector인 label_column(click)을 y변수로 선언
y_train = train_df[label_column]

#온라인 광고 클릭을 위한 테스트 데이터 준비
test_df = pd.read_csv('C:\\Users\\기백이 노트북\\Desktop\\pythonstudy\\AI_machinelearnning\\python_ML\\train.gz', compression='gzip', header=0, sep=",", quotechar='"', skiprows=(1,100000), nrows=100000)
test_df = test_df.drop(unused_columns, axis=1)
X_dict_test = list(test_df.drop(label_column, axis=1).T.to_dict().values())
y_test = test_df[label_column]

#DictVectorizer 객체 생성 후 앞에서 가공한 X_dict_train과 X_dict_test를 벡터로 변환 후 X_train, X_test로 각각 선언
vectorizer = DictVectorizer(sparse=True)
X_train = vectorizer.fit_transform(X_dict_train)
X_test = vectorizer.transform(X_dict_test)

#의사 결정 트리(CART 알고리즘을 이용) 학습 모델을 선언하고 gridsearchCV로 최적의 모델 탐색
#gridsearchCV에 사용할 hyperparams를 미리 선언
parameters = {'max_depth': [3, 10,None]}
#기본 decision tree model을 생성
decision_tree = DecisionTreeClassifier(criterion="gini",min_samples_split=30)
#gridsearch객체 선언 후 학습 ml 모델 입력, 하이퍼 파라미터 입력, scoring 방식 입력(roc_auc는 분류평가지표 중 하나로 ROC곡선 아래의 면적을 의미)
grid_search = GridSearchCV(decision_tree, parameters=parameters, n_jobs=-1, cv=3, scoring="roc_auc")
#선언한 gridsearch객체에 학습 데이터 적합
grid_search.fit(X_train, y_train)
#최적의 hyperparams를 출력
grid_search.best_params_
#grid_search 객체로 찾은 최적의 모델 인스턴스 선언
decision_tree_best = grid_search.best_estimator_