import logging
import numpy as np
from typing import Type
from dictionary import Dictionary
from pursuit import Pursuit

logging.basicConfig(level=logging.INFO)


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

    def sparse_coding(self, Y: np.ndarray):
        logging.info("Entering sparse coding stage...")
        if self.noise_gain and self.sigma:
            p = self.pursuit(self.dictionary, tol=(self.noise_gain * self.sigma))
        else:
            p = self.pursuit(self.dictionary, sparsity=self.sparsity)
        self.alphas = p.fit(Y)
        logging.info("Sparse coding stage ended.")

    def dictionary_update(self, Y: np.ndarray):
        # iterate rows
        D = self.dictionary.matrix
        n, K = D.shape
        print('Dictionary shape: {}'.format(D.shape))
        R = Y - D.dot(self.alphas)
        for k in range(K):
            logging.info("Updating column %s" % k)
            wk = np.nonzero(self.alphas[k, :])[0]
            if len(wk) == 0:
                continue
            Ri = R[:, wk] + D[:, k, None].dot(self.alphas[None, k, wk])
            U, s, Vh = np.linalg.svd(Ri)
            D[:, k] = U[:, 0]
            self.alphas[k, wk] = s[0] * Vh[0, :]
            R[:, wk] = Ri - D[:, k, None].dot(self.alphas[None, k, wk])
        self.dictionary = Dictionary(D)

    def fit(self, Y: np.ndarray, iter: int):
        for i in range(iter):
            logging.info("Start iteration %s" % (i + 1))
            self.sparse_coding(Y)
            self.dictionary_update(Y)
        return self.dictionary, self.alphas