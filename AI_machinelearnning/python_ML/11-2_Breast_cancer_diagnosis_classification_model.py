import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV



#@유방암 데이터를 dataframe 형태로 갖고오기
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
breast_cancer = pd.read_csv(data_url, sep=",",header=None)
print(breast_cancer.shape)

#@breast_cancer의 column 정보 입력하기
breast_cancer.columns = ["id_number", 
                         "clump_thickness", 
                         "unif_cell_size", 
                         "unif_cell_shape", 
                         "marg_adhesion", 
                         "single_epith_cell_size", 
                         "bare_nuclei", 
                         "bland_chromatin", 
                         "normal_nucleoli", 
                         "mitoses", 
                         "class"
                         ]
print(breast_cancer.head())
print(breast_cancer.info())
#각 열에 대한 정보를 출력한다.


#@데이터 전처리: 손실이 발생한 데이터(누락된 데이터)를 다른 데이터로 대체한다.
breast_cancer["bare_nuclei"]=breast_cancer["bare_nuclei"].replace('?',np.NAN)
null_count = breast_cancer.isnull().values.sum()
print(null_count)
#* null값의 개수를 출력

breast_cancer["bare_nuclei"] = breast_cancer["bare_nuclei"].fillna(breast_cancer["bare_nuclei"].value_counts().index[0])
print(type(breast_cancer["bare_nuclei"].value_counts().index[-4]))
#/*누락 값을 np.nan값(null)으로 바꾼뒤 NaN값을 breast_cancer["bare_nuclei"]의 최빈값으로 반환
#*fillna() 메소드는 np.nan값을 입력받은 값으로 모두 바꾸는 함수 
#*value_counts() 메소드느 원소의 빈도수를 serise 자료형으로 반환, 이때 serise의 인덱스는 각각의 원소, value는 빈도수이다.


#@Cancer_ind라는 column을 새로 만들고 class가 4인 경우(악성 종양인 경우) cancer_ind 가 1이 되도록 설정
breast_cancer["cancer_ind"] = 0
breast_cancer.loc[breast_cancer["class"]==4, "cancer_ind"] = 1
#*부울 배열과 마스킹 연산으로 위 코드 구현


#@X,y 훈련 데이터 생성 및 표준화 적용
X = breast_cancer.drop(["id_number", "class", "cancer_ind"], axis=1) 
#**input 특징행렬에는 정답이 포함되어있어서는 안되므로 정답 레이블및 불필요 레이블 제거
y = breast_cancer["cancer_ind"]
#**breast_cancer["cancer_ind"] 이것과 breast_cancer.cancer_ind 이거는 동일한 결과값을 출력한다.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#overfitting을 줄이기 위해 train data를 다시 train과 test로 나눠서 학습한다.

scaler = StandardScaler()
#데이터 표준화 객체 생성
print("훈련 데이터랑 테스트 데이터 출력")
print(X_train, X_test)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#훈련 데이터와 테스트 데이터를 데이터 표준화 객체를 통해 표준화


#@Machine Learning model class인 KNeighborsClassifier를 이용한 모델 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)


#@분류 모델의 confusion matrix, accuracy, AUC 계산 / 출력
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_pred, y_test)
print("정확도: {}".format(accuracy))

matrix = confusion_matrix(y_test, y_pred)
print("혼동행렬: {}".format(matrix))

rocauc_score = roc_auc_score(y_test, y_pred)
print("roc_auc_score: {}".format(rocauc_score))


#@GridSearch를 이용해서 최적의 hyperparameter를 탐색, 최적의 model을 만듬
params = {"n_neighbors":[1,2,3,4,5]}
grid_search = GridSearchCV(knn, params, n_jobs=1, cv = 7, scoring= "roc_auc")
grid_search.fit(X_train_scaled, y_train)

best_parameters = grid_search.best_params_
print("best_parameters:{}".format(best_parameters))

knn_best = grid_search.best_estimator_


#@최적의 모델의 정확도, 혼동행렬, ROC_AUC_SCORE 계산 및 출력
y_pred = knn_best.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("최적의 모델 정확도:{}".format(accuracy))

matrix = confusion_matrix(y_test, y_pred)
print("혼동행렬: {}".format(matrix))

rocauc_score = roc_auc_score(y_test, y_pred)
print("roc_auc_score: {}".format(rocauc_score))


#@모델 성능 시각화
fpr, tpr, unused_element = roc_curve(y_test, y_pred)
auc = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, "r-", label="KNN")
plt.plot([0, 1], [0, 1], "b--", label = "random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
print(auc)
plt.title("AUC={}".format(auc))
plt.show()