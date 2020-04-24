"""Proof of concept to draw clothoids.
"""

import numpy as np
import matplotlib.pyplot as plt
import typing
from clothoidlib import ClothoidCalculator, angle_between, fresnel

def pad(X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((X.shape[0], 1))])

def unpad(X: np.ndarray) -> np.ndarray:
    return X[:, :-1]

def compute_transformation_matrix(I: np.ndarray, O: np.ndarray) -> np.ndarray:
    A, *_ = np.linalg.lstsq(pad(I), pad(O), rcond=None)
    return A

def affine_transform(I: np.ndarray, A: np.ndarray):
    return unpad(pad(I) @ A)

def plot_clothoid(start: np.ndarray, intermediate: np.ndarray, goal: np.ndarray, calculator: ClothoidCalculator) -> np.ndarray:
    plt.plot(*list(zip(start, intermediate, goal, start)), c="k")

    params = calculator.lookup_points(start, intermediate, goal)
    print(params)

    gamma1, gamma2, alpha, beta, t1, t2 = params
    c0, c1, c2 = map(np.array, zip(*fresnel([np.zeros_like(t1), t1, t2])))

    P = np.array([goal, intermediate, start])
    C = np.array([c0, c1, c2])

    A = compute_transformation_matrix(C, P)
    plt.plot(*affine_transform(np.array(fresnel(np.linspace(0, t2, 200))).T, A).T, c="r")

if __name__ == "__main__":
    # Create calculator
    calculator = ClothoidCalculator()

    # Set up plot
    fig = plt.figure()
    plt.axis("equal")
    plt.plot([0, 1], [0, 1], c="w")
    plt.show(block=False)

    # Continually wait for user input
    while True:
        
        # Get user to add three points
        print("Click on three points on the plot, and a clothoid will be drawn.")
        points = np.array(plt.ginput(3, timeout=0))
        if points.shape != (3, 2):
            continue
        
        # Draw points
        plot_clothoid(*points, calculator)
        plt.show(block=False)
        fig.canvas.draw_idle()