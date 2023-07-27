"""
#@ 추천엔진
item 정보, user정보 등와 같은 많은 양의 데이터를 분석하고 데이터 마이닝 방식을 기반으로 연관 추천 제공
*USER에 맞는 추천을 하기위한 수학적 모델 or 목적 함수 개발이 핵심
가장 최선의 추천을 하는것이 목표


#@ 추천 엔진의 종류
*1. 협업 필터링 추천 시스템 : 사용자 선호도를 이용 -> 선택 가능한 가장 많은 집합들로부터 아이템을 필터링함
** 1) 사용자 기반 협업 필터링 : 사용자 취향을 고려해 추천 
                            **(1) 유사 선호도를 갖은 사용자 탐색
                            **(2) 활성 사용자 와 유사한 사용자가 제공한 아이템에 대한 등급을 참고 -> 활성 사용자에게 새로운 아이템 추천
** 2) 아이템 기반 협업 필터링 : 사용자가 선호하는 아이템과 유사한 제품을 추천
                            **(1) 아이템 선호도 기반으로 아이템 유사도 계산
                            **(2) 활성사용자가 과거 평가한 아이템과 가장 유사한 아지가 평가되지 않은 아이템을 탐색, 추천

*2. 콘텐츠 기반 추천 시스템 : 콘텐츠 정보를 활용 -> 아이템 속성과 그 속성에 대한 사용자 선호도를 통해 추천

*3. 하이브리드 추천 시스템 : 콘텐츠 기반 추천 시스템 + 협업 필터링 추천 시스템

*4. 상황 인식 추천 시스템 : 추천 항목을 계산하기 이전에 시간, 계절, 분위기, 장소 등 상황을 고려하는 유형의 추천 시스템


#@ 협업 필터링의 특징
1. 아이템에 대한 다른 다수의 선호도 집합을 활용
2. 유사성은 객관적인 성별, 나이 등이 아니라 사용자의 주관적인 평가에 의한 선호도를 의미
3. 아이템의 내용, 정보등을 반드시 알아야할 필요가 없음
4. 정보가 없는 새로운 아이템에 대한 사용자의 평가 예측이 가능
5. 시간의 흐름에 따라 고객의 선호도 변화를 감지, 유연성을 가짐

#@ 협업 엔지의 평가 
MSE(평균 제곱 오차), RMSE(평균 제곱근 오차)를 통해 성능을 평가한다.

* #@ Cosine Distance - 코사인 유사도
*코사인 유사도란?
Data 공간에서 Data를 벡터로 표현할때 서로 다른 두 벡터 A, B 사잇각을 통해 두 벡터가 얼마나 같은 방행으로 향하는지 측정 
-> A, B가 얼마나 유사한지를 계산한 값이다. 유사도는 -1~1의 값이고 0에 가까울수록 유사정도 줄어든다.
**(간단히 말해 A,B 사이 θ값에 대해 cos(θ)값을 구하고 값에 의미를 부여한 것이다.) 

/* #@ cosine_distances()의 의미
아래와 같은 행렬이 있다고 하자 
                        영화
            ______________________________
            |                             |  Row m 은 m번째 사람에 대한 정보들로 Row m 벡터는 m번째 사람의 영화 평가에 대한 정보를 원소로 갖고있다.
            |                             |  다시말해 영화라는 space에 m번째 사람에 대한 정보가 표준벡터 형식으로 되어있는 것이다.
        사람|                             |   Row n 은 n번째 사람에 대한 정보들로 Row n 벡터는 n번째 사람의 영화 평가에 대한 정보를 원소로 갖고이다.
            |                             |  
            .  .  .  .  .  .  .  .  .  .  .   Cosine_distances()는 m번째 사람과 n번째 사람 사이의 Cosine 유사도를 계산하여 m,n위치의 원소로 갖는 matrix를 
            .  .  .  .  .  .  .  .  .  .  .   반환하는 함수이다. 
            .  .  .  .  .  .  .  .  .  .  .   다시말해 Cosine_distances()은 (Row m * Row n)/(||Row m|| * ||Row n||)을 원소로 갖는 matrix를 반환하는 함수인 것이다.
            |                             |
            |                             |
            |                             |
            |_____________________________|

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors




#@movies rate data를 불러온다.
df = pd.read_csv(r"C:\Users\기백이 노트북\Desktop\pythonstudy\AI_machinelearnning\python_ML\ml-100k\u.data"
                 , sep='\t', header = None)
df.columns = ["user_id", "item_id", "rating", "timestamp"]
print(df.head())
print(df.shape)


#@ 평가별 사용자 수, 아이템별 사용자 수로 groupby를 이용해 묶는다.
standard_rating = df.groupby(["rating"])[["user_id"]].count()
standard_item_id = df.groupby(["item_id"])[["user_id"]].count()
print(standard_rating, standard_item_id)
#* groupby로 데이터프레임에서 그룹을 나눌때 groupby(["기준이 되는 column, 즉 index가 되는 column"])[["value가 되는 column"]]으로 묶는다.


#@평가행렬의 생성
n_users = df.user_id.unique().shape[0]
print("row는 n_users로 사용자 수:{}".format(n_users))
n_items = df.item_id.unique().shape[0]
print("column은 n_items로 평가된 영화 수:{}".format(n_items))
ratings = np.zeros((n_users, n_items))
print("0으로 채워진 평가 행렬로 평가 행렬의 각각의 row는 각각의 사용자가 평가한 모든 영화 평가 값이다." , ratings.shape)
# 유저별 평가 정보를 기록하기 위해 0으로 채워진 matrix를 만든다.

for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
print(type(ratings), ratings.shape)
print(ratings)
#* dataframe형태로 있는 사용자별 영화 평가를 np.array형태로 번환해서 평가행렬을 구성함 -> 평가행렬 = 특징행렬


#@평가행렬(특징행렬)을 훈련데이터와 테스트데이터로 나눈다.
ratings_train, ratings_test = train_test_split(ratings, test_size= 0.33, random_state= 42)
print(ratings_train.shape, ratings_test.shape)


#@사용자 간 유사도 행렬생성
similarity = cosine_distances(ratings_train)
print(similarity)

distances = 1-cosine_distances(ratings_train)
print(distances)


#@평가 예측 및 모델 성능 측정
user_pred = distances.dot(ratings_train)/np.array([np.abs(distances).sum(axis=1)]).T
#* dot()는 내적 연산을 하는 np.array의 멤버함수(method)이다.
#* T는 전치행렬로 바꾸는 np.array의 멤버함수(method)이다.
# 평가 예측 코드

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    #* flatten() 함수는 행렬로 존재하는 데이터를 맨 위에서부터 한 행씩 이어붙여 하나의 벡터로 펼치는 함수이다.
    return mean_squared_error(pred, actual)
# 평가 값 계산 함수 정의 및 선언

def rqrt_mean_squared_error_calculation(user_pred, data):
    rqrt_mean_squared_error = np.sqrt(get_mse(user_pred, data))
    print("RMSE계산 값:{:.5f}".format(rqrt_mean_squared_error))
    return rqrt_mean_squared_error

#@비슷한 n명을 찾는 비지도 방식의 이웃 검색
k = 5
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
#* K-nearest neighbor와 비슷하나 Nearest Neighbors는 비지도 학습이다. 다시말해 분류가 아니라 군집화이다.
#* cosine 유사도로 유사도 측정

neigh.fit(ratings_train)

top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)
#* kneighbors는 parameter에 따라 대상 간 k개의 유사도와 대상별(Row별) k개의 가장 유사도가 높은 대상을 원소로 갖는 두개의 2차 배열(matrix)를 반환한다.
print(top_k_distances.shape, top_k_users.shape)
print(top_k_users)
print(top_k_distances)


#@선택된 n명의 사용자들의 평가 가중치 합을 사용한 예측 및 모델의 성능 측정
user_pred_k = np.zeros(ratings_train.shape)

for i in range(ratings_train.shape[0]):
    user_pred_k[i,:] = top_k_distances[i].T.dot(ratings_train[top_k_users][i])/\
                        np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T

print(user_pred_k.shape, user_pred_k)
#예측 값 출력

train_data_MSE = get_mse(user_pred_k, ratings_train)
test_data_MSE = get_mse(user_pred_k, ratings_test)
train_data_RMSE = rqrt_mean_squared_error_calculation(user_pred_k, ratings_train)
test_data_RMSE = rqrt_mean_squared_error_calculation(user_pred_k, ratings_test)

print("train MSE계산 값:{:.5f}, test MSE계산 값:{:.5f}, train SMSE계산 값:{:.5f}, test SMSE계산 값:{:.5f}".format(
    train_data_MSE, test_data_MSE, train_data_RMSE, test_data_RMSE
))
"""
=============================================================================================================================================================================================================================================================================================================
/* 앞서 쓴 code의 가장 큰 단점은 비슷한 n명을 찾기위해 모든 사용자들에 대해 서로 간의 유사도를 계산한 것이다. 이는 사용자들이 매우 커지면 계산이 굉장히 오래걸린다.
*이러한 단점을 보완하여 비슷한 n명을 찾는 "비지도 학습" 모델인 NearestNeighbors를 사용할 것이다.
*KNeighborsClassifier는 metric의 default로 Lequysean Distance를 쓰며 이를 통해 데이터 포인트 간 거리를 측정한다.
*NearestNeighbors는 metric의 default로 cosine을 쓰며 이를 통해 Cosine Distance를 구한다.
"""


#@영화 수를 k로 사용해 영화 간 유사도 행렬 계산
k = ratings_train.shape[1] 
# 영화의 개수
neigh = neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
# 머신러닝 수행 모델 객체 선언

neigh.fit(ratings_train.T)
#* item들 간의 유사도를 계산하기 위해 transpose를 하여 item이 row vextor가 될 수 있도록 한다.

item_distances, _ = neigh.kneighbors(ratings_train.T, return_distance = True)
# 영화 간 유사도 행렬 계산 이때 '_'는 neigh_ind로 현 모델에서는 쓰임이 없는 값이다.


#@평가 예측 및 모델 성능 측정
item_pred = ratings_train.dot(item_distances)/np.array([np.abs(item_distances).sum(axis=1)])
print(item_pred.shape, item_pred)
# 평가 예측 출력

item_train_data_MSE = get_mse(item_pred, ratings_train)
item_test_data_MSE = get_mse(item_pred, ratings_test)
item_train_data_RMSE = rqrt_mean_squared_error_calculation(item_pred, ratings_train)
item_test_data_RMSE = rqrt_mean_squared_error_calculation(item_pred, ratings_test)

print("train MSE계산 값:{:.5f}, test MSE계산 값:{:.5f}, train RMSE계산 값:{:.5f}, test RMSE계산 값:{:.5f}".format(
    item_train_data_MSE, item_test_data_MSE, item_train_data_RMSE, item_test_data_RMSE
))
