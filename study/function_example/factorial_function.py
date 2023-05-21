'''순환 함수 형태: iterative form of factorial'''
def factorial1(n):
  result = 1
  for i in range(1,n+1):
    result = result*i
  return result

'''
재귀함수 형태: recursive form of factorial'''
def factorial2(n):
  if n==1:
    return 1
  else:
    return n*factorial2(n-1)
  

a = int(input("입력할 값을 쓰세요"))
f = factorial1(a)
print(f)