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

def get_angle_at(t):
    return np.pi * t**2 / 2

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

tmax = np.sqrt(3)
t_samples = np.linspace(0, tmax, 1000)
t1, t2 = np.stack(np.meshgrid(t_samples, t_samples), axis=-1).reshape(-1, 2).T
cond = (t1 < t2) & (t1 > 0)
t1 = t1[cond]
t2 = t2[cond]
gamma1, gamma2, *values = calculate_clothoid_parameters(t1, t2)
indices = np.array((gamma1, gamma2)).T
values = np.array(values).T
tree = KDTree(indices)

class ClothoidParameters(typing.NamedTuple):
    gamma1: np.ndarray
    gamma2: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    t1: np.ndarray
    t2: np.ndarray

def lookup_clothoid_parameters(gamma1, gamma2, k=1) -> ClothoidParameters:
    d, i = tree.query(np.stack([gamma1, gamma2], axis=-1), k)
    return ClothoidParameters(gamma1, gamma2, *values[i].T)

def lookup_clothoid_parameters_from_points(p0, p1, p2, k=1) -> ClothoidParameters:
    gamma1 = angle_between(p1-p2, p1-p0)
    gamma2 = angle_between(p2-p1, p2-p0)
    return lookup_clothoid_parameters(gamma1, gamma2, k=k)
