import os
import pickle
import random as rd

import numpy as np
import torch

rd.seed(42)

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class MinMaxScaler_torch:

    def __init__(self, min=None, max=None, device='cuda:0'):
        self.min = min
        self.max = max
        self.device = device

    def fit(self, data):
        self.min = torch.min(data)
        self.max = torch.max(data)

    def transform(self, data):
        _data = data.clone()
        return (_data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return (data * (self.max - self.min + 1e-8)) + self.min


class StandardScaler_torch:

    def __init__(self):
        self.means = 0
        self.stds = 0

    def fit(self, data):
        self.means = torch.mean(data, dim=0)
        self.stds = torch.std(data, dim=0)

    def transform(self, data):
        _data = data.clone()
        data_size = data.size()

        if len(data_size) > 2:
            _data = _data.reshape(-1, data_size[-1])

        _data = (_data - self.means) / (self.stds + 1e-8)

        if len(data_size) > 2:
            _data = _data.reshape(data.size())

        return _data

    def inverse_transform(self, data):
        data_size = data.size()
        if len(data_size) > 2:
            data = data.reshape(-1, data_size[-1])

        data = (data * (self.stds + 1e-8)) + self.means

        if len(data_size) > 2:
            data = data.reshape(data_size)

        return data


def granularity(data, k):
    if k == 1:
        return np.copy(data)
    else:
        newdata = [np.mean(data[i:i + k], axis=0) for i in range(0, data.shape[0], k)]
        newdata = np.asarray(newdata)
        print('new data: ', newdata.shape)
        return newdata


class PartialTrafficDataset(Dataset):

    def __init__(self, dataset, args):
        # save parameters
        self.args = args

        self.type = args.type
        self.out_seq_len = args.out_seq_len
        self.Xtopk = self.np2torch(dataset['Xtopk'])
        self.Ytopk = self.np2torch(dataset['Ytopk'])
        self.Yreal = self.np2torch(dataset['Yreal'])
        self.Xgt = self.np2torch(dataset['Xgt'])
        self.Ygt = self.np2torch(dataset['Ygt'])
        self.Topkindex = dataset['Topkindex']
        self.scaler_topk = dataset['Scaler_topk']

        self.nsample, self.len_x, self.nflows, self.nfeatures = self.Xtopk.shape

        # get valid start indices for sub-series
        self.indices = self.get_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x_top_k = self.Xtopk[t]
        y_top_k = self.Ytopk[t]
        y_real = self.Yreal[t]
        xgt = self.Xgt[t]
        ygt = self.Ygt[t]
        sample = {'x_top_k': x_top_k, 'y_top_k': y_top_k, 'x_gt': xgt, 'y_gt': ygt, 'y_real': y_real}
        return sample

    def transform(self, X):
        return self.scaler_topk.transform(X)

    def inverse_transform(self, X):
        return self.scaler_topk.inverse_transform(X)

    def np2torch(self, X):
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.to(self.args.device)
        return X

    def get_indices(self):
        indices = np.arange(self.nsample)
        return indices


def load_matlab_matrix(path, variable_name):
    X = loadmat(path)[variable_name]
    return X


def load_raw(args):
    # load ground truth
    path = args.datapath

    data_path = os.path.join(path, 'data/{}.mat'.format(args.dataset))
    X = load_matlab_matrix(data_path, 'X')
    if len(X.shape) > 2:
        X = np.reshape(X, newshape=(X.shape[0], -1))

    return X


def np2torch(X, device):
    X = torch.Tensor(X)
    if torch.cuda.is_available():
        X = X.to(device)
    return X


def data_preprocessing(data, topk_index, args, gen_times=5, scaler_top_k=None):
    n_timesteps, n_series = data.shape

    # original dataset with granularity k = 1
    oX = np.copy(data)
    oX = np2torch(oX, args.device)

    # Obtain data with different granularity k
    X = granularity(data, args.k)

    # Obtain dataset with topk flows
    X_top_k = np.copy(X[:, topk_index])

    X = np2torch(X, args.device)
    X_top_k = np2torch(X_top_k, args.device)

    # scaling data
    if scaler_top_k is None:
        scaler_top_k = StandardScaler_torch()
        scaler_top_k.fit(X_top_k)
    else:
        scaler_top_k = scaler_top_k

    X_scaled_top_k = scaler_top_k.transform(X_top_k)

    len_x = args.seq_len_x
    len_y = args.seq_len_y

    dataset = {'Xtopk': [], 'Ytopk': [], 'Xgt': [], 'Ygt': [], 'Yreal': [],
               'Topkindex': topk_index, 'Scaler_topk': scaler_top_k}

    skip = 4
    start_idx = 0
    for _ in range(gen_times):
        for t in range(start_idx, n_timesteps - len_x - len_y, len_x):
            x_topk = X_scaled_top_k[t:t + len_x]
            x_topk = x_topk.unsqueeze(dim=-1)  # add feature dim [seq_x, n, 1]

            y_topk = torch.max(X_top_k[t + len_x:t + len_x + len_y], dim=0)[0]
            y_topk = y_topk.reshape(1, -1)

            y_real = torch.max(X[t + len_x:t + len_x + len_y], dim=0)[0]
            y_real = y_real.reshape(1, -1)

            # Data for doing traffic engineering
            x_gt = oX[t * args.k:(t + len_x) * args.k]
            y_gt = oX[(t + len_x) * args.k: (t + len_x + len_y) * args.k]

            dataset['Xtopk'].append(x_topk)  # [sample, len_x, k, 1]
            dataset['Ytopk'].append(y_topk)  # [sample, 1, k]
            dataset['Yreal'].append(y_real)  # [sample, 1, k]
            dataset['Xgt'].append(x_gt)
            dataset['Ygt'].append(y_gt)

        start_idx = start_idx + skip

    dataset['Xtopk'] = torch.stack(dataset['Xtopk'], dim=0)
    dataset['Ytopk'] = torch.stack(dataset['Ytopk'], dim=0)
    dataset['Yreal'] = torch.stack(dataset['Yreal'], dim=0)
    dataset['Xgt'] = torch.stack(dataset['Xgt'], dim=0)
    dataset['Ygt'] = torch.stack(dataset['Ygt'], dim=0)

    dataset['Xtopk'] = dataset['Xtopk'].cpu().data.numpy()
    dataset['Ytopk'] = dataset['Ytopk'].cpu().data.numpy()
    dataset['Yreal'] = dataset['Yreal'].cpu().data.numpy()
    dataset['Xgt'] = dataset['Xgt'].cpu().data.numpy()
    dataset['Ygt'] = dataset['Ygt'].cpu().data.numpy()

    print('   Xtopk: ', dataset['Xtopk'].shape)
    print('   Ytopk: ', dataset['Ytopk'].shape)
    print('   Yreal: ', dataset['Yreal'].shape)
    print('   Xgt: ', dataset['Xgt'].shape)
    print('   Ygt: ', dataset['Ygt'].shape)
    print('   Topkindex: ', dataset['Topkindex'].shape)

    return dataset


def train_test_split(X):
    train_size = int(X.shape[0] * 0.5)
    val_size = int(X.shape[0] * 0.1)
    test_size = X.shape[0] - train_size - val_size

    if train_size >= 7000:
        train_size = 7000
    if val_size >= 1000:
        val_size = 1000

    if test_size >= 1000:
        test_size = 1000

    X_train = X[:train_size]

    X_val = X[train_size:val_size + train_size]

    X_test_list = []
    for i in range(10):
        X_test = X[val_size + train_size + test_size * i: val_size + train_size + test_size * (i + 1)]
        X_test_list.append(X_test)
        if val_size + train_size + test_size * (i + 1) >= X.shape[0]:
            break

    return X_train, X_val, X_test_list


def get_dataloader(args):
    # loading data
    X = load_raw(args)
    total_timesteps, total_series = X.shape

    stored_path = os.path.join(args.datapath, 'pdata/gwn_cs_{}_{}_{}_{}/'.format(args.dataset, args.seq_len_x,
                                                                                 args.seq_len_y, args.random_rate))
    if not os.path.exists(stored_path):
        os.makedirs(stored_path)

    saved_train_path = os.path.join(stored_path, 'train.pkl')
    saved_val_path = os.path.join(stored_path, 'val.pkl')
    saved_test_path = os.path.join(stored_path, 'test.pkl')
    if not os.path.exists(saved_train_path):
        train, val, test_list = train_test_split(X)
        means = np.mean(train, axis=0)
        top_k_index = np.argsort(means)[::-1]
        top_k_index = top_k_index[:int(args.random_rate * train.shape[1] / 100)]

        if args.top_k_random:
            top_k_index = np.random.randint(X.shape[1], size=top_k_index.shape[0])

        print('Data preprocessing: TRAINSET')
        trainset = data_preprocessing(data=train, topk_index=top_k_index, args=args, gen_times=10, scaler_top_k=None)
        train_scaler = trainset['Scaler_topk']
        with open(saved_train_path, 'wb') as fp:
            pickle.dump(trainset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        print('Data preprocessing: VALSET')
        valset = data_preprocessing(data=val, topk_index=top_k_index, args=args, gen_times=10,
                                    scaler_top_k=train_scaler)
        with open(saved_val_path, 'wb') as fp:
            pickle.dump(valset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        testset_list = []
        for i in range(len(test_list)):
            print('Data preprocessing: TESTSET {}'.format(i))
            testset = data_preprocessing(data=test_list[i], topk_index=top_k_index,
                                         args=args, gen_times=1, scaler_top_k=train_scaler)
            testset_list.append(testset)

        with open(saved_test_path, 'wb') as fp:
            pickle.dump(testset_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
    else:
        print('Load saved dataset from {}'.format(stored_path))
        with open(saved_train_path, 'rb') as fp:
            trainset = pickle.load(fp)
            fp.close()
        with open(saved_val_path, 'rb') as fp:
            valset = pickle.load(fp)
            fp.close()
        with open(saved_test_path, 'rb') as fp:
            testset_list = pickle.load(fp)
            fp.close()

    # Training set
    train_set = PartialTrafficDataset(trainset, args=args)
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True)

    # validation set
    val_set = PartialTrafficDataset(valset, args=args)
    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch_size,
                            shuffle=False)

    test_set = PartialTrafficDataset(testset_list[args.testset], args=args)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, total_timesteps, total_series
