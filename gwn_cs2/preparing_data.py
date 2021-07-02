import numpy as np
from scipy.io import loadmat, savemat

dataset = 'brain_tm'

data = loadmat('../../data/data/{}.mat'.format(dataset))['X']

data5 = [np.mean(data[i:i + 5], axis=0) for i in range(0, data.shape[0], 5)]
data5 = np.asarray(data5)
print('Brain5', data5.shape)
savemat('../../data/data/brain5_tm.mat', {'X': data5})

data10 = [np.mean(data[i:i + 10], axis=0) for i in range(0, data.shape[0], 10)]
data10 = np.asarray(data10)
print('Brain5', data10.shape)
savemat('../../data/data/brain5_tm.mat', {'X': data10})

dataset = 'abilene_tm'
data = loadmat('../../data/data/{}.mat'.format(dataset))['X']
data = data[:45000]

data15 = [np.mean(data[i:i + 3], axis=0) for i in range(0, data.shape[0], 3)]
data15 = np.asarray(data15)
print('Abilene15', data15.shape)
savemat('../../data/data/abilene15_tm.mat', {'X': data15})

data10 = [np.mean(data[i:i + 2], axis=0) for i in range(0, data.shape[0], 2)]
data10 = np.asarray(data10)
print('Abilene10', data10.shape)
savemat('../../data/data/abilene10_tm.mat', {'X': data10})
