import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#1차원~3차원 점에대한 데이터 생성
df_1d = pd.DataFrame(data = np.random.rand(60,1), columns =["1d_points"])
df_1d["height"] = 1

df_2d = pd.DataFrame(data = np.random.rand(60,2), columns= ["x","y"])

df_3d = pd.DataFrame(data = np.random.rand(60,3), columns=["x","y","z"])

#1,2차원에 대한 데이터 시각화
fig,ax = plt.subplots(2,1)
ax[0].scatter(df_1d["1d_points"], df_1d["height"])
ax[0].set_yticks([])
ax[0].set_xlabel(df_1d.columns[0])

ax[1].scatter(df_2d["x"], df_2d["y"])
ax[1].set_yticks([])
ax[1].set_xlabel(df_2d.columns[0])
ax[1].set_ylabel(df_2d.columns[1])

#3차원에 대한 데이터 시각화
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111, projection = "3d")
ax_2.scatter(df_3d["x"], df_3d["y"], df_3d["z"])
plt.show()