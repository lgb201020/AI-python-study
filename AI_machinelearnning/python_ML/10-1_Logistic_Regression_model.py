"""
@Logistic Regression
@Logistic Regression의 정의
 Logistic Regression: 베르누이 분포를 기반 예측기 모델이다.
 *베르누이 분포: 긍정 확률(P)와 부정 확률(1-P)를 갖는 랜덤 변수의 확률분포이다. 자언계의 일반적인 정규분포와 대칭되는,다른 구조의 분포이다.
 Logistic Regression는 베르누이 분포를 따르는 반응변수의 확률분포 범위가 0~1이다.
 /*Linear Regression은 "특징값의 일정한 변화->반응변수의 일정한 변화가 나타남" 이란 가정 기반의 모델, 반응변수가 확률일 경우 가정이 유효하지 않는다.

-일반화 선형 모델
 일반화 선형 모델: 연결함수를 사용하여 특징의 선형조합을 반응변수와 연결 --> 선형회귀의 가정이 유효하지 않은 경우를 해결 
                  따라서 특징의 선형조합을 정규분포를 따르지 않는 반응변수와 연계하려면 연결함수 필요
                 *(반응변수가 확률인 경우에도 선형 회귀 사용가능, 이게 로지스틱 회귀)

@Logistic Regression의 특징
 *1.로지스틱 회귀의 반응 변수 값: 긍정(양성) 클래스의 확률 = P
 -> 반응변수 값 >= 임계치 : 긍정(양성) 클래스 예측
 -> 반응변수 값 =< 임계치 : 부정(음성) 클래스 예측
 
/*2. Logistic function을 사용해서 반응변수를 특징의 linear combination 형태로 모델링함
     F(t) = 1 / (1 + exp(-t)) , 0 =< F(t) =< 1, t는 선형변수의 조합(b_0 + b*X)-------b_0는 상수항(초기값), b는 가중치

 *3. Logistic function = sigmoid function이다.
     원점을 기점으로 양성 클래스와 음성 클래스로 나뉨, 
     로지스틱 회귀는 이진분류 역활을 수행한다.

 *4. 로짓함수
     로짓함수는 로지스틱 함수의 역함수로 F(t)를 다시 특징의 조합으로 돌림
     로짓함수를 통해 선형방정식(로그와 승산비)를 얻을 수 있다.
     g(t) = ln(F(t)/(1-F(t))) = t = b_0 + b*X---------이때 F(t)/(1-F(t))가 승산비(odd rate)이다.
"""

#Sigmoid function 시각화

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))
#시그모이드 함수 정의    
    
z =np.linspace(-10,10,1000)
y = sigmoid(z)
#z, y축 설정

plt.plot(z,y)
plt.axhline(y=0, linestyle=":", color="black")
plt.axhline(y=0.5, linestyle=":", color="black")
plt.axhline(y=1, linestyle=":", color="black")
plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("sigmoid function")
plt.show()