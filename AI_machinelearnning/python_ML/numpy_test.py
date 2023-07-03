import numpy as np
np.array([1,4,2,5,3])
np.array([1,4,2,5,3],dtype=float)
np.array(range(i,i+3)for i in [1,4,2,5,3])
np.zeros(10)
np.ones((3,5))
np.full((2,5),9)
np.arange(0,10,2)
np.linspace(0,100,5,dtype=int)
np.random.random((3,3))
np.random.randint(0,10,(3,3))
np.random.normal(0,1,(3,3))

np.random.seed(0)

arr1 = np.random.randint(10,size=6)
arr2 = np.random.randint(10,size=(2,3))
print("arr1:\n%s",arr1)
print("ndim: %d, shape: %s, size: %d, dtype: %s" %(arr1.ndim, arr1.shape, arr1.size, arr1.dtype))

print("arr2:\n%s",arr2)
print("ndim: %d, shape: %s, size: %d, dtype: %s" %(arr2.ndim, arr2.shape, arr2.size, arr2.dtype))

arr1 = np.random.randint(10,size=6)
print(arr1)

arr1 = np.arange(10)
arr1[0:5:1]
arr1[:5:1]
arr1[:5:]
arr1[:5]
arr1[2:9:2]
arr1[2::2]
arr1[:-1]
arr1[-1:-11:-1]
arr1[5::-1]

arr2 = np.arange(12).reshape(-1,4)
arr2[:3,:4]
arr2[:,:]

arr2[:2,:3]
arr2[:2,2::-1]
arr2[1:,-1]
arr2[-1,:]
arr2[-1]

arr2 = arr1.reshape(-1,3)
np.concatenate([arr2,arr2],axis=0)
np.concatenate([arr2,arr2],axis=1)
np.vstack([arr2,arr2])
np.hstack([arr2,arr2])

np.random.seed(0)
arr2 = np.random.randint(1,10,(3,4))
s1,s2 = np.sum(arr2), arr2.sum()
s1_1, s2_1 = np.sum(arr2,axis=0), arr2.sum(axis=0) 
s1_2, s2_2 = np.sum(arr2,axis=1), arr2.sum(axis=1)

min1_1,max1_1 = np.min(arr2,axis=0), np.max(arr2,axis=0)
min1_2,max1_2 = arr2.min(axis=0), arr2.max(axis=0)
min2_1,max2_1 = np.min(arr2,axis=1), np.max(arr2,axis=1)
min2_2,max2_2 = arr2.min(axis=1), arr2.max(axis=1)

np.random.seed(1)
X = np.random.random((10,3))
Xmean = X.mean(axis=0)
Xcentered = X - Xmean

np.random.seed(2)
X = np.random.randint(1,10,(3,4))
np.sum((X>5)&(X<8))
np.sum((X>5)|(X<8))
X[(X>5)&(X<8)]

np.sum((X>5)&(X<8),axis=0)
np.sum((X>5)&(X<8),axis=1)

np.sum((X>5)|(X<8),axis=0)
np.sum((X>5)|(X<8),axis=1)

X = np.arange(12).reshape((3,4))

row = np.array([0,1,2])
col = np.array([1,2,3])
X[row]
X[:,col]
X[row,col]
X[row.reshape(-1,1),col]

X = np.zeros(12).reshape((3,4))
X[1,0] = 1
X[1,[1,3]] = 1
X[[0,2],[1,3]] = 2
X[0:3,[0,2]] = 3

np.random.seed(3)
x = np.array(np.random.randint(10, size=5))
np.sort(x)
x.sort()

idx = np.argsort(x)
x[idx]