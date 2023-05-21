"""n^3을 일반 항으로 하는 수열의 합"""
def sum_quad1(n):
  """변수 초기화"""
  partialSum = 0  
  for i in range(n+1):
    partialSum += i*i*i
  return partialSum

def sum_quad2(n):
  if n ==1:
    return 1
  else:
    return (n*n*n) + sum_quad2(n-1)
  
a = int(input("입력할 값을 쓰세요"))
s1 = sum_quad1(a)
s2 = sum_quad2(a)
print(s1,s2)