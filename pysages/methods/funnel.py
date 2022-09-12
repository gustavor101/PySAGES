import jax.numpy as np
from jax import grad
import numpy
import jax.numpy.linalg as linalg
import matplotlib.pyplot as plt


def cone(x, eje, Z_pos, Zcc, alpha, R, k):
    apex = Z_pos + (R / np.tan(alpha)) * eje
    x_fit = x - apex
    proj = np.dot(x_fit, eje) * eje
    x_perp = x_fit - proj
    F = np.dot(x_perp, x_perp) - np.dot(x_fit, eje) * np.tan(alpha) * np.dot(
        x_fit, eje
    ) * np.tan(alpha)
    # alternative calculation of F but it is not continous with cylinder calculation
    # norm = np.dot(x_fit, x_fit)
    # F = norm * np.power(np.cos(alpha), 2) - np.power(np.dot(x_fit, eje), 2)
    #    if F < 0.0:
    #        return 0.0
    #    else:
    #        return k*F
    return np.where(F < 0.0, 0.0, k * F)


def cylinder(x, eje, R, k):
    x_perp = x - np.dot(x, eje) * eje
    F = np.dot(x_perp, x_perp) - R * R
    #    if F < 0:
    #        return 0.0
    #    else:
    #        return k*F
    return np.where(F < 0.0, 0.0, k * F)


def funnel(x, A, B, Zcc, alpha, R, k):
    x = np.asarray(x)
    A_r = np.asarray(A)
    B_r = np.asarray(B)
    norm_eje = linalg.norm(B_r - A_r)
    eje = (B_r - A_r) / norm_eje
    Z_pos = A_r + Zcc * eje
    x_fit = x - A_r
    proj = np.dot(x_fit, eje)
    #    if proj < Zcc:
    #        return cone(x_fit, eje, Z_pos, Zcc, alpha, R, k)
    #    else:
    #        return cylinder(x_fit, eje, R, k)
    return np.where(
        proj < Zcc,
        cone(x_fit, eje, Z_pos, Zcc, alpha, R, k),
        cylinder(x_fit, eje, R, k),
    )


test_grad = grad(funnel)
x=np.array([0.,0.,10.])
print(test_grad(x, [1., 0., 0.], [3., 0., 0.], 1., np.pi / 4., 1., 10.))

lista = []
for i in range(0, 50):
    for j in range(24, 25):
        for k in range(0, 50):
            x = -2.0 + i * 4.0 / 25.0
            y = -4.0 + j * 4.0 / 25.0
            z = -2.0 + k * 4.0 / 50.0
            color = funnel(
                [x, y, z],
                A=[0, 0, 0],
                B=[1, 0, 0],
                Zcc=1.0,
                alpha=np.pi / 4.0,
                R=0.5,
                k=1000.0,
            )
            lista.append([x, y, z, color])
numpy.savetxt("color.txt", lista)
exit()
