import numpy as np


def get_phi(top_k_index):
    G = np.zeros((top_k_index.shape[0], 20))

    for i, j in enumerate(G):
        j[top_k_index[i]] = 1

    return G


top_k_index = np.array([1, 5, 7, 9, 15])

phi = get_phi(top_k_index)
print(phi.shape)
print(phi)
