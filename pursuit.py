import numpy as np
from sklearn.linear_model import orthogonal_mp
from sklearn.decomposition import SparseCoder
import os


class Pursuit:
    """
    Algorithms that inherit from this class are methods to solve problems of the like
    \min_A \| DA - Y \|_2 s.t. \|A\|_0 <= t.
    Here, D is a given dictionary of size (n x K)
    Y is a given matrix of size (n x N), where N is the number of samples
    The Pursuit will return a matrix A of size (K x N).
    """

    def __init__(self, dictionary, max_iter=False, tol=None, sparsity=None):
        self.D = dictionary
        self.max_iter = max_iter
        self.tol = tol
        self.sparsity = sparsity
        if (self.tol is None and self.sparsity is None) or (self.tol is not None and self.sparsity is not None):
            raise ValueError("blub")
        self.data = None
        self.alphas = []

    def fit(self, Y):
        return [], self.alphas


class MatchingPursuit(Pursuit):
    """
    Standard Matching Pursuit
    """

    def fit(self, Y):
        # analyze shape of Y
        data_n = Y.shape[0]
        if len(Y.shape) == 1:
            self.data = np.array([Y])
        elif len(Y.shape) == 2:
            self.data = Y
        else:
            raise ValueError("Input must be a vector or a matrix.")

        # analyze dimensions
        n, K = self.D.shape
        if not n == data_n:
            raise ValueError("Dimension mismatch: %s != %s" % (n, data_n))

        for y in self.data.T:
            # temporary values
            coeffs = np.zeros(K)
            residual = y

            # iterate
            i = 0
            if self.max_iter:
                m = self.max_iter
            else:
                m = np.inf

            finished = False

            while not finished:
                if i >= m:
                    break
                inner = np.dot(self.D.T, residual)
                gamma = int(np.argmax(np.abs(inner)))
                alpha = inner[gamma]
                residual = residual - alpha * self.D[:, gamma]
                if np.isclose(alpha, 0):
                    break
                coeffs[gamma] += alpha
                coeffs[coeffs < 0.0] = 0.0
                i += 1
                if self.sparsity:
                    finished = np.count_nonzero(coeffs) >= self.sparsity
                #                     finished = np.sum(coeffs >= 0) >= self.sparsity
                else:
                    finished = (np.linalg.norm(residual) ** 2 < n * self.tol ** 2) or i >= n / 2
            self.alphas.append(coeffs)
        return np.transpose(self.alphas)


class OrthogonalMatchingPursuit(Pursuit):
    """
    Wrapper for orthogonal_mp from scikit-learn
    """

    def fit(self, Y):
        return orthogonal_mp(self.D, Y, n_nonzero_coefs=self.sparsity,
                             tol=self.tol, precompute=True)


class ThresholdingPursuit(Pursuit):
    """
    Thresholding pursuit
    """

    def __init__(self, dictionary, sparsity):
        super().__init__(dictionary, sparsity=sparsity)

    def fit(self, Y):
        gammas = np.zeros((Y.shape[1], self.D.shape[1]))
        inners = np.abs(np.matmul(self.D.T, Y))
        idx = np.argsort(-inners.T)[:self.sparsity, :self.sparsity]
        gammas.T[idx] = inners[idx]
        return gammas.T


def sparse_coding(ZT, phiT, psiT):
    """
    As SparseCoding: X = Code*Dict | As in the paper: Zt = phi*psi*S
    X:(n, N_F); Code:(n, N_C); Dict:(N_C, N_F)
    --> Z.T = S.T * psi.T * phi.T
    Z.T:(n, k); S.T:(n, N_C) | psi.T:(N_C, N_F)=Dict, phi.T:(N_F, k)
    k: number of topk flows
    N_C=N_F: total number of flows
    n: total number timesteps/samples

    ------
    Input:
    - ZT:(n, k)
    - phiT:(N_F, k)
    - psiT: (N_C, N_F)
    return:
    - Shat:(n, N_C)
    """
    # analyze shape of Y
    if len(ZT.shape) == 1:
        data = np.array([ZT])
    elif len(ZT.shape) == 2:
        data = np.copy(ZT)
    else:
        raise ValueError("Input must be a vector or a matrix.")

    # analyze dimensions
    N_C, N_F = psiT.shape
    assert N_C == N_F
    k = phiT.shape[1]
    A = np.dot(psiT, phiT)  # shape (N_C, k)
    assert k == ZT.shape[1]

    coder = SparseCoder(dictionary=A, transform_algorithm='lasso_lars',
                        transform_alpha=1e-10, positive_code=True, n_jobs=os.cpu_count(),
                        transform_max_iter=1000)

    Shat = coder.transform(ZT)
    return Shat  # shape(n, N_C)
