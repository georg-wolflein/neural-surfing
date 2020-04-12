import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel as _fresnel
from scipy.spatial import KDTree
import pandas as pd
from functools import partial
import typing
from pathlib import Path

def fresnel(*args):
    return list(reversed(_fresnel(*args)))

tmax = np.sqrt(3)

def get_angle_at(t):
    return np.pi * t**2 / 2
def dy_dx(t):
    return np.tan(get_angle_at(t))

def gradient_vector_at(t):
    xs, ys = fresnel(t)
    dys = dy_dx(t)
    dxs = np.ones_like(ys)
    vecs = np.stack((dxs, dys), axis=-1)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True) * 8
    xy = np.stack((xs, ys), axis=-1)
    return np.stack((xy - vecs, xy + vecs), axis=-2)

CLOTHOID_CACHE = dict()


t_samples = np.linspace(0, tmax, 500)

def angle_between(v1, v2):
    v1 /= np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=-1, keepdims=True)
    v1 = np.expand_dims(v1, axis=-2)
    v2 = np.expand_dims(v2, axis=-1)
    return np.arccos(np.matmul(v1, v2)).reshape(v1.shape[:-2])

def calculate_clothoid_parameters(t1, t2):
    ts = np.stack((np.zeros_like(t1), t1, t2), axis=0)

    p0, p1, p2 = np.stack(fresnel(ts), axis=-1)

    gamma1 = angle_between(p1-p0, p1-p2)
    gamma2 = angle_between(p2-p0, p2-p1)
    theta = get_angle_at(t2) # angle at end
    omega = np.arctan(p1[..., 1] / p1[..., 0])
    beta = omega + np.pi - gamma1 - gamma2
    alpha = theta - beta

    return gamma1, gamma2, alpha, beta, t1, t2

t1, t2 = np.stack(np.meshgrid(t_samples, t_samples), axis=-1).reshape(-1, 2).T
cond = (t1 < t2) & (t1 > 0)
t1 = t1[cond]
t2 = t2[cond]
gamma1, gamma2, *values = calculate_clothoid_parameters(t1, t2)
indices = np.array((gamma1, gamma2)).T
values = np.array(values).T
tree = KDTree(indices)

def lookup_parameters(gamma1, gamma2, k=1):
    d, i = tree.query(np.stack([gamma1, gamma2], axis=-1), k)
    return values[i]

def pad(X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

def unpad(X):
    return X[:,:-1]

def compute_transformation_matrix(I, O):
    A, *_ = np.linalg.lstsq(pad(I), pad(O), rcond=None)
    return A

def affine_transform(I, A):
    return unpad(pad(I) @ A)

def plot_clothoid_given_points(p0, p1, p2):
    p0, p1, p2 = np.array([p0, p1, p2], dtype=np.float32)
    plt.plot(*list(zip(p2, p1, p0, p2)), c="k")

    gamma1 = angle_between(p1-p2, p1-p0)
    gamma2 = angle_between(p2-p1, p2-p0)

    alpha, beta, t1, t2 = lookup_parameters(gamma1, gamma2)

    print("alpha:", alpha)
    print("beta: ", beta)

    c0, c1, c2 = map(np.array, zip(*fresnel([np.zeros_like(t1), t1, t2])))

    P = np.array([p0, p1, p2])
    C = np.array([c0, c1, c2])

    A = compute_transformation_matrix(C, P)
    plt.plot(*affine_transform(np.array(fresnel(np.linspace(0, t2, 200))).T, A).T, c="r")

fig = plt.figure()
plt.axis("equal")
plt.plot([0, 1], [0, 1], c="w")
plt.show(block=False)
while True:

    print("Click on three points on the plot, and a clothoid will be drawn.")
    points = np.array(plt.ginput(3, timeout=0))
    if points.shape != (3, 2):
        continue

    p2, p1, p0 = points
    plot_clothoid_given_points(p0, p1, p2)
    plt.show(block=False)


    fig.canvas.draw_idle()