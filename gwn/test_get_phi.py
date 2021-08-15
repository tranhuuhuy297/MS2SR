import pickle
from scipy.io import loadmat, savemat
import numpy as np
import math
from dictionary import DCTDictionary
from ksvd import KSVD
from pursuit import MatchingPursuit, sparse_coding

a = np.random.randint(0, 1000, (1000, 1000))
# b = torch.Tensor(a).to('cuda:0')
#
# with open('test.pkl', 'wb') as fp:
#     pickle.dump(b, fp, protocol=pickle.HIGHEST_PROTOCOL)
#     fp.close()

seq_len_x = 12
seq_len_y = 12
mon_rate = 1


def get_psi(X, samples=4000):
    X = X[:samples]
    X_temp = np.array([np.max(X[seq_len_x + i:
                                seq_len_x + i + seq_len_y], axis=0) for i in
                       range(samples - seq_len_x - seq_len_y)])

    N = X.shape[1]
    D = np.zeros(shape=(N, N))

    psiT, ST = KSVD(D, MatchingPursuit, sparsity=int(mon_rate / 100 * X.shape[1])).fit(X_temp)
    return psiT, ST


# file = 'abilene_tm_2_12_12_psi.pkl'
# psi_save_path = '/home/anle/data/cs/saved_psi/{}'.format(file)
#
# with open(psi_save_path, 'rb') as fp:
#     obj = pickle.load(fp)
#     fp.close()
# psi = obj['psi']
# alpha = obj['alpha']
#
# alpha = alpha.T
# print(np.where(alpha[alpha < 0]))


datapath = '/home/anle/data/data/abilene_tm.mat'
data = loadmat(datapath)['X']
data = np.reshape(data, newshape=(data.shape[0], -1))

psi, alpha = get_psi(X=data)
print(psi.shape)
