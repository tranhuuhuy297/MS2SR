import logging
from typing import Type

import numpy as np

from dictionary import Dictionary
from pursuit import Pursuit
from sklearn.decomposition import SparseCoder
logging.basicConfig(level=logging.INFO)
from sklearn.decomposition import MiniBatchDictionaryLearning
import os
class KSVD:
    def __init__(self, dictionary: Dictionary, pursuit: Type[Pursuit], sparsity: int, noise_gain=None, sigma=None):
        self.dictionary = Dictionary(dictionary.matrix)
        self.alphas = None
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

    # def sparse_coding(self, Y: np.ndarray):
    #     logging.info("Entering sparse coding stage...")
    #     if self.noise_gain and self.sigma:
    #         p = self.pursuit(dictionary=self.dictionary, tol=(self.noise_gain * self.sigma))
    #     else:
    #         p = self.pursuit(dictionary=self.dictionary, sparsity=self.sparsity)
    #     self.alphas = p.fit(Y)
    #     logging.info("Sparse coding stage ended.")

    def sparse_coding(self, Y: np.ndarray):
        logging.info("Entering sparse coding stage...")
        coder = SparseCoder(dictionary=self.dictionary.matrix.T, transform_algorithm='lasso_lars',
                            transform_alpha=1e-10,
                            positive_code=True, n_jobs=os.cpu_count() - 4, )
        self.alphas = coder.transform(Y)
        self.alphas = self.alphas.T
        logging.info("Sparse coding stage ended.")

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
    def dictionary_update(self, X: np.ndarray):
        D = self.dictionary.matrix
        n, K = D.shape
        dict_learner = MiniBatchDictionaryLearning(n_components=K, transform_algorithm='lasso_lars', random_state=42,
                                                   fit_algorithm='cd', dict_init=D.T, positive_code=True, n_jobs=6)
        alphas = dict_learner.fit_transform(X.T)

        newdict = dict_learner.inner_stats_[0]
        self.dictionary = Dictionary(newdict.T)
        self.alphas = alphas.T

    def fit(self, X: np.ndarray, iter: int):
        # for i in range(iter):
        #     logging.info("Start iteration %s" % (i + 1))
        # self.sparse_coding(Y.T)
        self.dictionary_update(X)

        return self.dictionary, self.alphas
