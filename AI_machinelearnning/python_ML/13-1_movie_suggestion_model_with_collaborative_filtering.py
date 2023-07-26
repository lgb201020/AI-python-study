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

def sqrt_mean_squared_error_calculation(user_pred, data):
    sqrt_mean_squared_error = np.sqrt(get_mse(user_pred, data))
    print("SMSE계산 값:{:.5f}".format(sqrt_mean_squared_error))
    return sqrt_mean_squared_error

#@비슷한 n명을 찾는 비지도 방식의 이웃 검색
k = 5
neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
#* K-nearest neighbor와 비슷하나 Nearest Neighbors는 비지도 학습이다. 다시말해 분류가 아니라 군집화이다.
#* cosine 유사도로 유사도 측정

neigh.fit(ratings_train)

top_k_distances, top_k_users = neigh.kneighbors(ratings_train, return_distance=True)
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
train_data_SMSE = sqrt_mean_squared_error_calculation(user_pred_k, ratings_train)
test_data_SMSE = sqrt_mean_squared_error_calculation(user_pred_k, ratings_test)

print("train MSE계산 값:{:.5f}, test MSE계산 값:{:.5f}, train SMSE계산 값:{:.5f}, test SMSE계산 값:{:.5f}".format(
    train_data_MSE, test_data_MSE, train_data_SMSE, test_data_SMSE
))


#@영화 수를 k로 사용해 영화 간 유사도 행렬 계산
k = ratings_train.shape[1] 
# 영화의 개수
neigh = neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
# 머신러닝 수행 모델 객체 선언

neigh.fit(ratings_train.T)

item_distances, _ = neigh.kneighbors(ratings_train.T, return_distance = True)
# 영화 간 유사도 행렬 계산 이때 '_'는 neigh_ind로 현 모델에서는 쓰임이 없는 값이다.


#@평가 예측 및 모델 성능 측정
item_pred = ratings_train.dot(item_distances)/np.array([np.abs(item_distances).sum(axis=1)])
print(item_pred.shape, item_pred)
# 평가 예측 출력

item_train_data_MSE = get_mse(item_pred, ratings_train)
item_test_data_MSE = get_mse(item_pred, ratings_test)
item_train_data_SMSE = sqrt_mean_squared_error_calculation(item_pred, ratings_train)
item_test_data_SMSE = sqrt_mean_squared_error_calculation(item_pred, ratings_test)

print("train MSE계산 값:{:.5f}, test MSE계산 값:{:.5f}, train SMSE계산 값:{:.5f}, test SMSE계산 값:{:.5f}".format(
    item_train_data_MSE, item_test_data_MSE, item_train_data_SMSE, item_test_data_SMSE
))
