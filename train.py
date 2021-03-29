import pickle
import numpy as np

from scipy.io import loadmat

from ksvd import KSVD
from pursuit import MatchingPursuit
from dictionary import RandomDictionary

# X = loadmat('/home/aiotlab/huyth/data/data/abilene_tm.mat')['X']

# D = RandomDictionary(12, 12)
# X_temp = X[:100].T

# psi, S = KSVD(D, MatchingPursuit, 15).fit(X_temp, 5)


def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)

print(largest_indices(np.array([2,3,4,5,1]), 2))