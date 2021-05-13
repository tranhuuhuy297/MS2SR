import os
import random as rd

import numpy as np
import torch

rd.seed(42)

from .util import largest_indices

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class MinMaxScaler_torch():

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


class StandardScaler_torch():

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
        return data
    else:
        newdata = [np.mean(data[i:i + k], axis=0) for i in range(0, data.shape[0], k)]
        newdata = np.asarray(newdata)
        print('new data: ', newdata.shape)
        return newdata


class TrafficDataset(Dataset):

    def __init__(self, X, args, scaler=None, scaler_top_k=None, top_k_index=None):
        # save parameters
        self.args = args

        self.type = args.type
        self.out_seq_len = args.out_seq_len
        self.trunk = args.trunk

        self.oX = np.copy(X)
        self.oX = self.np2torch(self.oX)

        # granularity
        self.k = args.k
        X = granularity(X, args.k)

        # get top k flows
        self.top_k_index = top_k_index

        # data train to get psi
        self.X_top_k = np.copy(X[:, self.top_k_index])
        self.X = self.np2torch(X)
        self.X_top_k = self.np2torch(self.X_top_k)

        self.n_timeslots, self.n_series = self.X.shape

        # learn scaler
        if scaler is None:
            self.scaler = StandardScaler_torch()
            self.scaler.fit(self.X)
        else:
            self.scaler = scaler

        if scaler_top_k is None:
            self.scaler_top_k = StandardScaler_torch()
            self.scaler_top_k.fit(self.X_top_k)
        else:
            self.scaler_top_k = scaler_top_k

        # transform if needed and convert to torch
        self.X_scaled = self.scaler.transform(self.X)
        self.X_scaled_top_k = self.scaler_top_k.transform(self.X_top_k)

        if args.tod:
            self.tod = self.get_tod()

        if args.ma:
            self.ma = self.get_ma()

        if args.mx:
            self.mx = self.get_mx()

        # get valid start indices for sub-series
        self.indices = self.get_indices()

        if torch.isnan(self.X).any():
            raise ValueError('Data has Nan')

    def get_tod(self):
        tod = torch.arange(self.n_timeslots, device=self.args.device)
        tod = (tod % self.args.day_size) * 1.0 / self.args.day_size
        tod = tod.repeat(self.n_series, 1).transpose(1, 0)  # (n_timeslot, nseries)
        return tod

    def get_ma(self):
        ma = torch.zeros_like(self.X_scaled, device=self.args.device)
        for i in range(self.n_timeslots):
            if i <= self.args.seq_len_x:
                ma[i] = self.X_scaled[i]
            else:
                ma[i] = torch.mean(self.X_scaled[i - self.args.seq_len_x:i], dim=0)

        return ma

    def get_mx(self):
        mx = torch.zeros_like(self.X_scaled, device=self.args.device)
        for i in range(self.n_timeslots):
            if i == 0:
                mx[i] = self.X_scaled[i]
            elif 0 < i <= self.args.seq_len_x:
                mx[i] = torch.max(self.X_scaled[0:i], dim=0)[0]
            else:
                mx[i] = torch.max(self.X_scaled[i - self.args.seq_len_x:i], dim=0)[0]
        return mx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.X_scaled[t:t + self.args.seq_len_x]  # step: t-> t + seq_x
        # xgt = self.X[t:t + self.args.seq_len_x]  # step: t-> t + seq_x
        xgt = self.oX[t * self.k:(t + self.args.seq_len_x) * self.k]
        x = x.unsqueeze(dim=-1)  # add feature dim [seq_x, n, 1]

        # top k
        x_top_k = self.X_scaled_top_k[t:t + self.args.seq_len_x]  # step: t-> t + seq_x
        xgt_top_k = self.X_top_k[t:t + self.args.seq_len_x]  # step: t-> t + seq_x
        x_top_k = x_top_k.unsqueeze(dim=-1)  # add feature dim [seq_x, n, 1]

        y = torch.max(self.X[t + self.args.seq_len_x:
                             t + self.args.seq_len_x + self.args.seq_len_y], dim=0)[0]

        y = y.reshape(1, -1)

        # top k
        y_top_k = torch.max(self.X_top_k[t + self.args.seq_len_x:
                                         t + self.args.seq_len_x + self.args.seq_len_y], dim=0)[0]

        y_top_k = y_top_k.reshape(1, -1)

        if self.args.tod:
            tod = self.tod[t:t + self.args.seq_len_x]
            tod = tod.unsqueeze(dim=-1)  # [seq_x, n, 1]
            x = torch.cat([x, tod], dim=-1)  # [seq_x, n, +1]

        if self.args.ma:
            ma = self.ma[t:t + self.args.seq_len_x]
            ma = ma.unsqueeze(dim=-1)  # [seq_x, n, 1]
            x = torch.cat([x, ma], dim=-1)  # [seq_x, n, +1]

        if self.args.mx:
            mx = self.mx[t:t + self.args.seq_len_x]
            mx = mx.unsqueeze(dim=-1)  # [seq_x, n, 1]
            x = torch.cat([x, mx], dim=-1)  # [seq_x, n, +1]

        # ground truth data for doing traffic engineering
        y_gt = self.oX[(t + self.args.seq_len_x) * self.k:
                       (t + self.args.seq_len_x + self.args.seq_len_y) * self.k]

        y_gt_top_k = self.X_top_k[t + self.args.seq_len_x: t + self.args.seq_len_x + self.args.seq_len_y]

        sample = {'x': x, 'y': y, 'x_gt': xgt, 'y_gt': y_gt,
                  'x_top_k': x_top_k, 'y_top_k': y_top_k,
                  'x_gt_top_k': xgt_top_k, 'y_gt_top_k': y_gt_top_k}
        return sample

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def np2torch(self, X):
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.to(self.args.device)
        return X

    def get_indices(self):
        T, D = self.X.shape
        indices = np.arange(T - self.args.seq_len_x - self.args.seq_len_y)
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


def train_test_split(X):
    train_size = int(X.shape[0] * 0.7)
    val_size = int(X.shape[0] * 0.1)

    X_train = X[:train_size]

    X_val = X[train_size:val_size + train_size]

    X_test = X[val_size + train_size:]

    return X_train, X_val, X_test


def get_dataloader(args):
    # loading data
    X = load_raw(args)

    if X.shape[0] > 10000:
        _size = 10000
    else:
        _size = X.shape[0]

    X = X[:_size]

    train, val, test = train_test_split(X)

    if train.shape[0] > 5000:
        train = train[-5000:]

    random_time_step = rd.randint(0, len(train))
    # get top k biggest

    top_k_index = largest_indices(train[random_time_step], int(args.random_rate / 100 * train.shape[1]))
    top_k_index = np.sort(top_k_index)[0]

    # Training set
    train_set = TrafficDataset(train, args=args, scaler=None, top_k_index=top_k_index)
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True)

    # validation set
    val_set = TrafficDataset(val, args=args, scaler=train_set.scaler, top_k_index=top_k_index)
    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch_size,
                            shuffle=False)

    test_set = TrafficDataset(test, args=args, scaler=train_set.scaler, top_k_index=top_k_index)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, top_k_index
