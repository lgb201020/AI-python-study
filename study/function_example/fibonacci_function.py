"""순환 함수 형태: iterative form of fibonacci"""
def fibonacci1(n):
  if n<2:
    return n
  else:
    current = 1
    last = 0
    for i in range(2,n+1):
       tmp = current
       current = current + last
       last = tmp
    return current
  
'''재귀함수 형태: recursive form of fibonacci'''
def fibonacci2(n):
  
  if n == 0:
    return 0

  elif n<3:
    return 1

  else:
    return fibonacci2(n-1)+fibonacci2(n-2)