import os
import sys

sys.path.append('..')

import time
import math
import models
import torch
import utils
from tqdm import trange
from routing import *

from dictionary import RandomDictionary
from ksvd import KSVD
from pursuit import MatchingPursuit

import warnings

def get_psi(args, iterator=100):
    X = utils.load_raw(args)

    X = X[:10000, :]

    X_temp = np.array([np.max(X[args.seq_len_x + i: \
        args.seq_len_x + i + args.seq_len_y], axis=0) for i in range(10000 - args.seq_len_x - args.seq_len_y)]).T

    size_D = int(math.sqrt(X.shape[1]))

    D = RandomDictionary(size_D, size_D)

    psi, S = KSVD(D, MatchingPursuit, int(args.random_rate/100 * X.shape[1])).fit(X_temp, iterator)

    return psi, S

if __name__ == "__main__":
    args = utils.get_args()
    psi, S = get_psi(args, 10)
    print(S.shape)
    import numpy as np
    np.save('s.npy', S)