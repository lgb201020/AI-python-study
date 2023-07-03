#learning curve
from sklearn.model_selection import validation_curve
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

np.random.seed(1)
X = np.random.rand(40, 1)**2
y = (10-1./(X.ravel()+0.1))+np.random.randn(40)

fig, ax = plt.subplots(1,2,figsize=(16,6))
#두개의 도면과 두개의 축을 만든다.
for i, degree in enumerate([2,9]): 
    n, train_lc, val_lc = learning_curve(make_pipeline(PolynomialFeatures(degree = degree), LinearRegression()), X, y, cv =7, train_sizes=np.linspace(0.1,1,25))
    #learning_curve()는 전체 training 크기, training의 learning curve, validation의 learning curve를 반환-> learning curve의 매개변수 지정
    #이때 train size의 default값은 train_sizes=np.linspace(0.1,1,5)->X축에 25개의 구간이 만들어 진다.
    ax[i].plot(n,np.mean(train_lc, 1),"b-",label = "training score")
    ax[i].plot(n,np.mean(val_lc, 1),"r-",label = "validation score")
    #mean함수에서 결과값을 벡터로 출력하려면 mean(train_lc, 1)과 같이 행에 대한 정보도 써야한다.
    ax[i].hlines(np.mean([train_lc[-1],val_lc[-1]]), n[0], n[-1], color = "gray", linestyle = "dashed")
    ax[i].set(xlim = (n[0],n[-1]), ylim = (0,1), xlabel = "train sizes", ylabel = "score")
    ax[i].set_title("degree = {}".format(degree), size = 14)
    ax[i].legend(loc = "best")
    
plt.show()