import numpy as np
import matplotlib.pyplot as plt

x, y, z, w = np.genfromtxt("color.txt", unpack=True)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
img = ax.scatter(x, y, z, c=w, cmap=plt.viridis())
fig.colorbar(img)
plt.show()
exit()
