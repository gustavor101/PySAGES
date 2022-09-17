import numpy as np
import matplotlib.pyplot as plt

x, y, z, u, v, w = np.genfromtxt("colorvec.txt", unpack=True)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
img = ax.quiver(x, y, z, u,v,w, pivot='middle')
plt.show()
exit()
