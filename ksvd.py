import logging
import numpy as np
from typing import Type
from dictionary import Dictionary
from pursuit import Pursuit

from sklearn.decomposition import SparseCoder
from sklearn.decomposition import DictionaryLearning

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
        sparse = SparseCoder(dictionary=self.dictionary.matrix, transform_algorithm='lasso_lars',
                            positive_code=True)
        self.alphas = sparse.transform(Y)
        self.alphas = self.alphas.T
        logging.info("Sparse coding stage ended.")

    def dictionary_update(self, Y: np.ndarray):
        D = self.dictionary.matrix
        n, K = D.shape
        dict_learner = DictionaryLearning(n_components=K, transform_algorithm='lasso_lars', random_state=42,
                                          fit_algorithm='cd', dict_init=D, positive_code=True,
                                          max_iter=1000, verbose=True)
        alphas = dict_learner.fit_transform(Y.T)

        newdict = dict_learner.components_
        self.dictionary = Dictionary(newdict.T)
        self.alphas = alphas.T

    def fit(self, Y: np.ndarray, iter: int):
        self.dictionary_update(Y)
        return self.dictionary, self.alphas