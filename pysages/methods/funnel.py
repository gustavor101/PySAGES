import jax.numpy as np
from jax import grad
import numpy
import jax.numpy.linalg as linalg
import matplotlib.pyplot as plt
from pysages.colvars.funnel-cv import center, create_matrot, quaternion_matrix

def cone(x, eje, Z_pos, alpha, R, k):
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

def rotation_AB(pos_ref, references, weights, A, B):
    com_prot = center(pos_ref, weights)
    _, rotmat = linalg.eigh(quaternion_matrix(pos_ref, references, weights))
    quat = rotmat[:3]
    A_rot = np.matmul(create_matrot(quat), A - com_prot)
    B_rot = np.matmul(create_matrot(quat), B - com_prot)
    return A_rot + com_prot, B_rot + com_prot

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
    #        return cone(x_fit, eje, Z_pos, alpha, R, k)
    #    else:
    #        return cylinder(x_fit, eje, R, k)
    return np.where(
        proj < Zcc,
        cone(x_fit, eje, Z_pos, alpha, R, k),
        cylinder(x_fit, eje, R, k),
    )

def external_funnel(data, indices, references,weights_ligand, weights_protein, A, B, Zcc, alpha, R, k):
    indices_ligand=indices[0]
    indices_protein= indices[1]
    pos=data.positions[:,:3]
    ids= data.indices
    pos_ligand = center(pos[ids[indices_ligand]],weights_ligand)
    pos_protein= pos[ids[indices_protein]]
    A_rot, B_rot = rotation_AB(pos_protein,references,weights_protein, A, B)
    return funnel(pos_ligand, A_rot, B_rot, Zcc, alpha, R, k)
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
lista = []
for i in range(0, 5):
    for j in range(0, 5):
        for k in range(0, 5):
            x = -2.0 + i * 4.0 / 2.5
            y = -4.0 + j * 4.0 / 2.5
            z = -2.0 + k * 4.0 / 5.0
            color = test_grad(
                [x, y, z],
                A=[0, 0, 0],
                B=[1, 0, 0],
                Zcc=1.0,
                alpha=np.pi / 4.0,
                R=0.5,
                k=0.1,
            )
            lista.append([x, y, z, -color[0], -color[1], -color[2]])
numpy.savetxt("colorvec.txt", lista)
exit()
