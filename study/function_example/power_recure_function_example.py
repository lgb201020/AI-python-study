"""math 라이브러리에 있는 method 사용해서 거듭제곱 계산"""
import math as m
m.pow(2,3)

'''순환 함수 형태:iterative form of 거듭제곱'''
def power_recur1(x,n):
  
  x_n = x
  for i in range(n):
    x_n =x_n*x
    
  return x_n


'''재귀함수 형태: recursive form of 거듭제곱'''
def power_recur2(x,n):
  if n == 0:
    return 1
  elif n%2 == 0:
    return power_recur2(x*x,n//2)
  else:
    return x*power_recur2(x*x,n//2)
