import numpy as np
from scipy.special import fresnel as _fresnel
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import euclidean_distances
import typing

fresnel = lambda x: tuple(reversed(_fresnel(x)))

class ClothoidParameters(typing.NamedTuple):
    """A named tuple for storing clothoid parameters
    """

    gamma1: np.ndarray
    gamma2: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    t2: np.ndarray

def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Utility function to calculate the angle between two vectors.

    This method is vectorized, so when supplying matrices, they will be interpreted as collections of vectors.

    Arguments:
        v1 {np.ndarray} -- the first vector
        v2 {np.ndarray} -- the second vector

    Returns:
        np.ndarray -- the calculated angle(s)
    """
    v1 /= np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=-1, keepdims=True)
    v1 = np.expand_dims(v1, axis=-2)
    v2 = np.expand_dims(v2, axis=-1)
    return np.arccos(np.matmul(v1, v2)).reshape(v1.shape[:-2])

def cartesian_product(*arrays):
    # Compute the Cartesian product of arrays
    # Thanks to https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def argmin_where(arr, mask):
    subset_index = np.argmin(arr[mask])
    parent_index = np.arange(arr.shape[0])[mask][subset_index]
    return parent_index

def compute_clothoid_table(t_samples: np.ndarray) -> ClothoidParameters:
    """Calculate the clothoid parameters given t1 and t2.

    This method is vectorized, so when supplying vectors for the angles, they will be interpreted as collections of scalars.

    Arguments:
        t1 {np.ndarray} -- the value of t1
        t2 {np.ndarray} -- the value of t2

    Returns:
        ClothoidParameters -- the calculated parameters
    """

    # Create a matrix of all possible combinations of t1 and t2 values
    t1, t2 = np.stack(np.meshgrid(t_samples, t_samples), axis=-1).reshape(-1, 2).T

    # Discard the rows where t1 < t2
    mask = (t1 < t2) & (t1 > 0)
    t1 = t1[mask]
    t2 = t2[mask]

    ts = np.stack((np.zeros_like(t1), t1, t2), axis=0)

    p0, p1, p2 = np.stack(fresnel(ts), axis=-1)

    gamma1 = angle_between(p1-p0, p1-p2)
    gamma2 = angle_between(p2-p0, p2-p1)
    theta = np.pi * t2**2 / 2 # angle at end
    omega = np.arctan(p1[..., 1] / p1[..., 0])
    beta = omega + np.pi - gamma1 - gamma2
    alpha = theta - beta

    # Calculate t0
    lengths = np.linalg.norm(p2 - p1, axis=-1)
    t0 = np.zeros_like(t1)
    print(len(t0))
    for i, (length, current_t1) in enumerate(zip(lengths, t1)):
        if i % 1000 == 0:
            print(i)
        mask = t2 == current_t1
        masked_lengths = lengths[mask]
        if masked_lengths.shape[0] == 0:
            t0[i] = 0
        else:
            masked_index = np.argmin(np.abs(masked_lengths - length))
            t0[i] = np.arange(t1.shape[0])[mask][masked_index]

    return ClothoidParameters(gamma1, gamma2, alpha, beta, t0, t1, t2)


class ClothoidCalculator:
    """Fast and efficient computation of clothoids.
    """

    def __init__(self, samples: float = 1000, t_max: float = np.sqrt(3)):
        t_samples = np.linspace(0, t_max, samples)

        # Calculate clothoid parameters for each t1 and t2
        gamma1, gamma2, *values = compute_clothoid_table(t_samples)

        # Construct kd-tree
        indices = np.array((gamma1, gamma2)).T
        self._values = np.array(values).T
        self._tree = KDTree(indices)
    
    def lookup_angles(self, gamma1: np.ndarray, gamma2: np.ndarray) -> ClothoidParameters:
        """Lookup clothoid parameters by providing the values of gamma1 and gamma2. 

        This method is vectorized, so when supplying arrays to gamma1 and gamma2, the parameters will be in array form too.

        Arguments:
            gamma1 {np.ndarray} -- the value of the first angle (radians)
            gamma2 {np.ndarray} -- the value of the second angle (radians)

        Returns:
            ClothoidParameters -- the calculated parameters
        """

        # Query the kd-tree
        d, i = self._tree.query(np.stack([gamma1, gamma2], axis=-1), k=1)
        result = gamma1, gamma2, *self._values[i].T
        return ClothoidParameters(*map(np.array, result))

    def lookup_points(self, start: np.ndarray, intermediate: np.ndarray, goal: np.ndarray) -> ClothoidParameters:
        """Lookup clothoid parameters by providing a triple of points. 

        This method is vectorized, so when supplying arrays of points, the parameters will be in array form too.

        Arguments:
            start {np.ndarray} -- the starting point
            intermediate {np.ndarray} -- the intermediate sample point
            goal {np.ndarray} -- the goal point

        Returns:
            ClothoidParameters -- the calculated parameters
        """

        # Calculate gamma1 and gamma2
        p0, p1, p2 = goal, intermediate, start
        gamma1 = angle_between(p1-p2, p1-p0)
        gamma2 = angle_between(p2-p1, p2-p0)

        # Perform lookup
        return self.lookup_angles(gamma1, gamma2)
