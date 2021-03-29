import numpy as np


def random_dictionary(n, K, normalized=True, seed=None):
    """
    Build a random dictionary matrix with K = n
    Args:
        n: square of signal dimension
        K: square of desired dictionary atoms
        normalized: If true, columns will be l2-normalized
        seed: Random seed

    Returns:
        Random dictionary
    """
    if seed:
        np.random.seed(seed)
    H = np.random.rand(n, K) * 255
    if normalized:
        for k in range(K):
            H[:, k] *= 1 / np.linalg.norm(H[:, k])
    return np.kron(H, H)
