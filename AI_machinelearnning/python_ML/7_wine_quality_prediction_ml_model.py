import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import displot
import time

red_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";", header = 0)
red_wine["type"] = "red"
white_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";", header = 0)
white_wine["type"] = "white"

#두 데이터 프레임을 행 방향으로 합침
wine = pd.concat([red_wine, white_wine], axis=0)

#컬럼명 공백에 "_"로 채움
wine.columns=wine.columns.str.replace(" ", "_")

#wine dataframe의 수치형 변수들의 요약 통계량을 describe()함수를 통해 출력
wine.describe()
wine.quality.describe()

#요약 통계량을 오름차순으로 확인 및 빈도수 계산
sorted(wine.quality.unique())
wine.quality.value_counts()

#와인 종류별 품질의 기술 통계량
wine.groupby("type")["quality"].describe()

#와인 종류별 품질의 사분위수
wine.groupby("type")["quality"].quantile([0.25, 0.5, 0.75, 1]).unstack("type")

import seaborn as sns
import matplotlib.pyplot as plt

# 와인 종류별 품질 데이터 가져오기
red_q = wine.loc[wine["type"] == "red", "quality"]
white_q = wine.loc[wine["type"] == "white", "quality"]

sns.set_style("darkgrid")

# histplot을 사용하여 히스토그램 그리기
sns.histplot(red_q, kde=False, color="red", label="Red wine", stat="density")
sns.histplot(white_q, kde=False, color="blue", label="White wine", stat="density")

plt.title("Distribution of Quality of Red and White Wines")
plt.xlabel("Quality Score")
plt.ylabel("Density")
plt.legend()

#와인 종류별 품질 차이의 통계적 유의성 검정
wine.groupby("type")["quality"].aggregate(["std", "mean"])

#통계적 검정에 사용되는 라이브러리 중 하나
import statsmodels.api as sm

#t-검정 수행
t_stat, p_value, df = sm.stats.ttest_ind(red_q, white_q)
t = "t-stat:{:.3f}, p-value:{:.4f}".format(t_stat,p_value)

#상관 분석: 변수들 사이에 상관 계수 계산 corr() 함수는 모든 변수들 사이의 상관관계를 계산하는 함수 
#이때 wine에는 red, white의 string 타입의 데이터가 있으므로 문자열 데이터를 제거 후 숫자 데이터만 따로 선택하여 상관계수 계산
numerical_columns = wine.select_dtypes(include=[np.number])
wine_corr = numerical_columns.corr()
"""
상관계수 R은 -1 <= R <= 1 범위의 값을 가짐
1. 1에 가까우면 양의 상관관계
2. -1에 가까우면 음의 상관관계
3. 0에 가까우면 상관관계가 없음
"""

#양의 상관관계
positive_corr = wine_corr.loc[wine_corr["quality"] > 0, "quality"]

#음의 상관관계
negative_corr = wine_corr.loc[wine_corr["quality"] < 0, "quality"]                                                                                                                                                                                                                                                                                                                         

#산점도 행렬

#산점도행렬을 그리기 위해 red와 white와인 데이터를 분리
red_sample = wine.loc[wine["type"]=="red", :]
white_sample = wine.loc[wine["type"]=="white", :]

#각 집단별 추출할 샘플 인덱스 200개 생성
red_idx = np.random.choice(red_sample.index, replace= True, size = 200)
white_idx = np.random.choice(white_sample.index, replace= True, size = 200)

#생성한 인덱스를 통해 샘플 데이터의 인덱스를 추출
wine_sample = pd.concat([red_sample.loc[red_idx,], white_sample.loc[white_idx,]], axis=0)

#산점도 시각화
sns.set_style("dark")
sns.pairplot(wine_sample, vars=["quality", "alcohol", "residual_sugar"], kind="reg", plot_kws={"ci":False, "x_jitter":0.25, "y_jitter":0.25},
             diag_kind="hist", diag_kws={"bins":10, "alpha":0.6}, hue="type", palette=dict(red="red",white="blue",markers=["o","s"]));
plt.show()
#히스토그램과 스캐터플롯 2개의 그림을 그려줌
#kind="reg"로 산점도에 회귀선 표시