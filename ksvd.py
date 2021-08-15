import logging
from typing import Type

import numpy as np

from dictionary import Dictionary
from pursuit import Pursuit
from sklearn.decomposition import SparseCoder
logging.basicConfig(level=logging.INFO)
from sklearn.decomposition import DictionaryLearning
import os
class KSVD:
    def __init__(self, dictionary: Dictionary, pursuit: Type[Pursuit], sparsity: int, noise_gain=None, sigma=None,
                 verbose=False):
        self.dictionary = Dictionary(dictionary.matrix)
        self.code = None
        self.pursuit = pursuit
        self.sparsity = sparsity
        self.noise_gain = noise_gain
        self.sigma = sigma
        self.original_image = None
        self.sparsity_values = []
        self.mses = []
        self.ssims = []
        self.psnrs = []
        self.iter = None
        self.verbose = verbose

    # def sparse_coding(self, Y: np.ndarray):
    #     logging.info("Entering sparse coding stage...")
    #     if self.noise_gain and self.sigma:
    #         p = self.pursuit(dictionary=self.dictionary, tol=(self.noise_gain * self.sigma))
    #     else:
    #         p = self.pursuit(dictionary=self.dictionary, sparsity=self.sparsity)
    #     self.alphas = p.fit(Y)
    #     logging.info("Sparse coding stage ended.")

    # def dictionary_update(self, Y: np.ndarray):
    #     # iterate rows
    #     D = self.dictionary.matrix
    #     n, K = D.shape
    #     R = Y - D.dot(self.alphas)
    #     for k in range(K):
    #         logging.info("Updating column %s" % k)
    #         wk = np.nonzero(self.alphas[k, :])[0]
    #         if len(wk) == 0:
    #             continue
    #         Ri = R[:, wk] + D[:, k, None].dot(self.alphas[None, k, wk])
    #         U, s, Vh = np.linalg.svd(Ri)
    #         D[:, k] = U[:, 0]
    #         self.alphas[k, wk] = s[0] * Vh[0, :]
    #         R[:, wk] = Ri - D[:, k, None].dot(self.alphas[None, k, wk])
    #     self.dictionary = Dictionary(D)
    #
    def fit(self, X: np.ndarray):
        """
        X: shape (n, N_F)
        dictionary: shape (N_C, N_F)
        ----
        X: (n, N_F)
        SparseCoding: X(n, N_F) = Code(T, N_C)*Dict(N_C, N_F)
        In the paper: X(N_F, n) = psi(N_F, N_C)*S(N_C, n)
        --> self.dictionary = Dict(N_C, N_F) = (psi).T
        --> self.code = Code = (S)^T
        """
        D = self.dictionary.matrix
        N_C, N_F = D.shape
        dict_learner = DictionaryLearning(n_components=N_C, transform_algorithm='lasso_lars', random_state=42,
                                          fit_algorithm='cd', dict_init=D, positive_code=True,
                                          n_jobs=os.cpu_count() - 4,
                                          max_iter=1000, verbose=self.verbose)
        self.code = dict_learner.fit_transform(X)
        self.dictionary = Dictionary(dict_learner.components_)
        return self.dictionary, self.code
