import pickle

import numpy as np

a = np.random.randint(0, 1000, (1000, 1000))
# b = torch.Tensor(a).to('cuda:0')
#
# with open('test.pkl', 'wb') as fp:
#     pickle.dump(b, fp, protocol=pickle.HIGHEST_PROTOCOL)
#     fp.close()

with open('test.pkl', 'rb') as fp:
    c = pickle.load(fp)
    fp.close()

print(c.get_device())
print(c.is_cuda)
