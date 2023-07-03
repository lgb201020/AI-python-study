import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib_inline

plt.style.available
plt.style.use("seaborn-whitegrid")

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), "-", linewidth=1, label="sin(x)")
plt.plot(x, np.cos(x), "--", linewidth=1, label="cos(x)")
plt.xlim(0, 10)
plt.ylim(-5, 5)
plt.title("sin & cos graph")
plt.xlabel("x")
plt.legend()
plt.show()

with plt.style.context("ggplot"):
    plt.plot(x, np.sin(x), "-", linewidth=1, label="sin(x)")
    plt.plot(x, np.cos(x), "--", linewidth=1, label="cos(x)")
    plt.xlim(0, 10)
    plt.ylim(-5, 5)
    plt.title("sin & cos graph")
    plt.xlabel("x")
    plt.legend()
    plt.show()

fig = plt.figure()
ax = plt.axes()
x = np.linspace(0,10,100)
ax.plot(x, np.sin(x), "-" ,linewidth = 1, label = "sin(x)")
ax.plot(x, np.cos(x), "k--" ,linewidth = 1, label = "cos(x)")
ax.set_xlim([0,10])
ax.set_ylim([-1,1])
ax.set_title("sin & cos graph")
ax.set_xlabel("X")
ax.legend()
plt.show()

fig,ax = plt.subplot(2,1)
ax[0].plot(x,np.sin(x),"b-",linewidth = 1, title = "sin(x)")
ax[0].set_title("sin & cos graph")
ax[0].set(xlim =[0,10], ylim=[-1,1])
ax[0].legend()
ax[1].plot(x,np.cos(x),'g-',linewidth = 1, title = "cos(x)")
ax[1].set(xlim=[0,10],ylim=[-1,1])
ax[1].legend()
