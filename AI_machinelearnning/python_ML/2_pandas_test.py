import numpy as np
import pandas as pd 
import seaborn as sns

#series object
data = pd.Series(np.linspace(0,1,num = 5))
data.values
data.index
data[1]
data[2:4]
data[[2,4]]
list(data.keys())
list(data.items())

data.index = ["a","b","c","d","e"]
data.index
data.loc["a"]
data.loc["a":"c"]
data.loc[["a","c"]]
data.loc[data > 0.7]



