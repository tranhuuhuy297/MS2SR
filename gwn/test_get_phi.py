from dictionary import DCTDictionary
from ksvd import KSVD
from pursuit import MatchingPursuit, Solver_l0
import pickle

import numpy as np

a = np.random.randint(0, 1000, (1000, 1000))
# b = torch.Tensor(a).to('cuda:0')
#
# with open('test.pkl', 'wb') as fp:
#     pickle.dump(b, fp, protocol=pickle.HIGHEST_PROTOCOL)
#     fp.close()

file = 'abilene_tm_2_12_12_psi.pkl'
psi_save_path = '/home/anle/data/cs/saved_psi/{}'.format(file)

with open(psi_save_path, 'rb') as fp:
    obj = pickle.load(fp)
    fp.close()
psi = obj['psi']
alpha = obj['alpha']

alpha = alpha.T
print(np.where(alpha[alpha < 0]))
