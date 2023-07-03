import numpy as np

rs = np.random.RandomState(10)
x = 10 * rs.rand(5)
y = 2 * x - 1 * rs.rand(5)
x.shape, y.shape

X = x.reshape(-1,1)
X.shape

import seaborn as sns
iris = sns.load_dataset("iris")
iris.info

iris.head()