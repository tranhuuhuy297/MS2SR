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


class StandardScaler_torch_all:

    def __init__(self):
        self.means = 0
        self.stds = 0

    def fit(self, data):
        self.means = torch.mean(data)
        self.stds = torch.std(data)

    def transform(self, data):
        _data = data.clone()
        data_size = data.size()

        if len(data_size) > 2:
            _data = _data.reshape(-1, data_size[-1])

        _data = (_data - self.means) / (self.stds + 1e-12)

        if len(data_size) > 2:
            _data = _data.reshape(data.size())

        return _data

    def inverse_transform(self, data):
        data_size = data.size()
        if len(data_size) > 2:
            data = data.reshape(-1, data_size[-1])

        data = (data * (self.stds + 1e-12)) + self.means

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
        self.Topkindex = self.np2torch(dataset['Topkindex'])
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
        topk_index = self.Topkindex[t]
        sample = {'x_top_k': x_top_k, 'y_top_k': y_top_k, 'x_gt': xgt, 'y_gt': ygt, 'y_real': y_real,
                  'topk_index': topk_index}
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


def get_tod(n_timeslots, n_series, day_size, device):
    tod = torch.arange(n_timeslots, device=device)
    tod = (tod % day_size) * 1.0 / day_size
    tod = tod.repeat(n_series, 1).transpose(1, 0)  # (n_timeslot, nseries)
    return tod


def get_ma(X, seq_len_x, n_timeslots, device):
    ma = torch.zeros_like(X, device=device)
    for i in range(n_timeslots):
        if i <= seq_len_x:
            ma[i] = X[i]
        else:
            ma[i] = torch.mean(X[i - seq_len_x:i], dim=0)

    return ma


def get_mx(X, seq_len_x, n_timeslots, device):
    mx = torch.zeros_like(X, device=device)
    for i in range(n_timeslots):
        if i == 0:
            mx[i] = X[i]
        elif 0 < i <= seq_len_x:
            mx[i] = torch.max(X[0:i], dim=0)[0]
        else:
            mx[i] = torch.max(X[i - seq_len_x:i], dim=0)[0]
    return mx


def topk_gt(Xscaled, X, oX, t, n_mflows, args):
    # obtain exact topk flow using original data X
    means = np.mean(X[t:t + args.seq_len_x], axis=0)
    topk_idx = np.argsort(means)[::-1]
    topk_idx = topk_idx[:n_mflows]

    # x_topk from scaled data and topk_idx
    x_topk = Xscaled[t:t + args.seq_len_x, topk_idx]
    x_topk = np.expand_dims(x_topk, axis=-1)  # [len_x, k, 1]

    # y_topk and yreal from original data
    y_topk = np.max(X[t + args.seq_len_x: t + args.seq_len_x + args.seq_len_y, topk_idx], keepdims=True,
                    axis=0)  # [1, k] max of each flow in next routing cycle

    y_real = np.max(X[t + args.seq_len_x:t + args.seq_len_x + args.seq_len_y], keepdims=True, axis=0)

    # Data for doing traffic engineering
    x_gt = oX[t * args.k:(t + args.seq_len_x) * args.k]  # Original X, in case of scaling data
    y_gt = oX[(t + args.seq_len_x) * args.k: (
                                                     t + args.seq_len_x + args.seq_len_y) * args.k]  # Original Y, in case of scaling data

    return x_topk, y_topk, y_real, x_gt, y_gt, topk_idx


def topk_rand(Xscaled, X, oX, t, n_mflows, args):
    # obtain exact topk flow using original data X
    topk_idx = [-1] * n_mflows
    for i in range(n_mflows):
        while True:
            rand_idx = np.random.randint(0, n_mflows)
            if rand_idx not in topk_idx:
                topk_idx[i] = rand_idx
                break
    topk_idx = np.asarray(topk_idx)

    # x_topk from scaled data and topk_idx
    x_topk = Xscaled[t:t + args.seq_len_x, topk_idx]
    x_topk = np.expand_dims(x_topk, axis=-1)  # [len_x, k, 1]

    # y_topk and yreal from original data
    y_topk = np.max(X[t + args.seq_len_x: t + args.seq_len_x + args.seq_len_y, topk_idx], keepdims=True,
                    axis=0)  # [1, k] max of each flow in next routing cycle

    y_real = np.max(X[t + args.seq_len_x:t + args.seq_len_x + args.seq_len_y], keepdims=True, axis=0)

    # Data for doing traffic engineering
    x_gt = oX[t * args.k:(t + args.seq_len_x) * args.k]  # Original X, in case of scaling data
    y_gt = oX[(t + args.seq_len_x) * args.k: (
                                                     t + args.seq_len_x + args.seq_len_y) * args.k]  # Original Y, in case of scaling data

    return x_topk, y_topk, y_real, x_gt, y_gt, topk_idx


def topk_prand(Xscaled, X, oX, t, n_mflows, topk_idx, n_rand_flows, args):
    # obtain exact topk flow using original data X
    n_timesteps, n_series = oX.shape
    if topk_idx.size == 0:
        means = np.mean(X[t:t + args.seq_len_x], axis=0)
        topk_idx = np.argsort(means)[::-1]
        topk_idx = topk_idx[:n_mflows]
    else:
        for i in range(n_mflows - n_rand_flows, n_mflows, 1):
            while True:
                rand_idx = np.random.randint(0, n_series)
                if rand_idx not in topk_idx:
                    topk_idx[i] = rand_idx
                    break

    # x_topk from scaled data and topk_idx
    x_topk = Xscaled[t:t + args.seq_len_x, topk_idx]
    x_topk = np.expand_dims(x_topk, axis=-1)  # [len_x, k, 1]

    # y_topk and yreal from original data
    y_topk = np.max(X[t + args.seq_len_x: t + args.seq_len_x + args.seq_len_y, topk_idx], keepdims=True,
                    axis=0)  # [1, k] max of each flow in next routing cycle

    y_real = np.max(X[t + args.seq_len_x:t + args.seq_len_x + args.seq_len_y], keepdims=True, axis=0)

    # Data for doing traffic engineering
    x_gt = oX[t * args.k:(t + args.seq_len_x) * args.k]  # Original X, in case of scaling data
    y_gt = oX[(t + args.seq_len_x) * args.k: (
                                                     t + args.seq_len_x + args.seq_len_y) * args.k]  # Original Y, in case of scaling data

    return x_topk, y_topk, y_real, x_gt, y_gt, topk_idx


def data_preprocessing(data, args, gen_times=5, scaler_top_k=None):
    n_timesteps, n_series = data.shape

    oX = np.copy(data)
    # oX = np2torch(oX, args.device)

    X = granularity(data, args.k)
    X = np2torch(X, args.device)

    # scaling data
    if scaler_top_k is None:
        scaler_top_k = StandardScaler_torch_all()
        scaler_top_k.fit(X)
    else:
        scaler_top_k = scaler_top_k

    X_scaled = scaler_top_k.transform(X)  # dtype torch

    X = X.cpu().data.numpy()  # dtype numpy
    X_scaled = X_scaled.cpu().data.numpy()  # numpy

    n_mflows = int(args.mon_rate * n_series / 100)
    n_rand_flows = int(50 * n_mflows / 100)
    len_x = args.seq_len_x
    len_y = args.seq_len_y

    dataset = {'Xtopk': [], 'Ytopk': [], 'Xgt': [], 'Ygt': [], 'Yreal': [],
               'Topkindex': [], 'Scaler_topk': scaler_top_k}

    skip = 4
    start_idx = 0
    for _ in range(gen_times):
        topk_idx = np.empty(0)
        for t in range(start_idx, n_timesteps - len_x - len_y, len_x):

            if args.fs == 'gt':
                x_topk, y_topk, y_real, x_gt, y_gt, topk_idx = topk_gt(Xscaled=X_scaled, X=X, oX=oX,
                                                                       t=t, n_mflows=n_mflows, args=args)
            elif args.fs == 'prand':
                x_topk, y_topk, y_real, x_gt, y_gt, topk_idx = topk_prand(Xscaled=X_scaled, X=X, oX=oX,
                                                                          t=t, n_mflows=n_mflows, args=args,
                                                                          topk_idx=topk_idx, n_rand_flows=n_rand_flows)
            elif args.fs == 'rand':
                x_topk, y_topk, y_real, x_gt, y_gt, topk_idx = topk_rand(Xscaled=X_scaled, X=X, oX=oX,
                                                                         t=t, n_mflows=n_mflows, args=args)
            else:
                raise RuntimeError('No flow selection!')

            dataset['Xtopk'].append(x_topk)  # [sample, len_x, k, 1]
            dataset['Ytopk'].append(y_topk)  # [sample, 1, k]
            dataset['Yreal'].append(y_real)  # [sample, 1, k]
            dataset['Xgt'].append(x_gt)
            dataset['Ygt'].append(y_gt)
            dataset['Topkindex'].append(np.copy(topk_idx))

        start_idx = start_idx + skip

    dataset['Xtopk'] = np.stack(dataset['Xtopk'], axis=0)
    dataset['Ytopk'] = np.stack(dataset['Ytopk'], axis=0)
    dataset['Yreal'] = np.stack(dataset['Yreal'], axis=0)
    dataset['Xgt'] = np.stack(dataset['Xgt'], axis=0)
    dataset['Ygt'] = np.stack(dataset['Ygt'], axis=0)
    dataset['Topkindex'] = np.stack(dataset['Topkindex'], axis=0)

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
    X = load_raw(args)
    total_timesteps, total_series = X.shape
    # loading data

    stored_path = os.path.join(args.datapath, 'cs_data/csp_{}_{}_{}_{}_{}/'.format(args.dataset, args.seq_len_x,
                                                                                   args.seq_len_y,
                                                                                   args.mon_rate,
                                                                                   args.fs))
    if not os.path.exists(stored_path):
        os.makedirs(stored_path)

    saved_train_path = os.path.join(stored_path, 'train.pkl')
    saved_val_path = os.path.join(stored_path, 'val.pkl')
    saved_test_path = os.path.join(stored_path, 'test.pkl')
    if not os.path.exists(saved_train_path):

        if args.fs == 'rand':
            gentime_train = 10
            gentime_test = 10
        else:
            gentime_train = 10
            gentime_test = 1

        train, val, test_list = train_test_split(X)
        print('Data preprocessing: TRAINSET')
        trainset = data_preprocessing(train, args, gen_times=gentime_train)
        train_scaler = trainset['Scaler_topk']
        with open(saved_train_path, 'wb') as fp:
            pickle.dump(trainset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        print('Data preprocessing: VALSET')
        valset = data_preprocessing(val, args, gen_times=gentime_train, scaler_top_k=train_scaler)
        with open(saved_val_path, 'wb') as fp:
            pickle.dump(valset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        testset_list = []
        for i in range(len(test_list)):
            print('Data preprocessing: TESTSET {}'.format(i))
            testset = data_preprocessing(test_list[i], args, gen_times=gentime_test, scaler_top_k=train_scaler)
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