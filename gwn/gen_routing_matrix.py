import sys

import numpy as np

sys.path.append('..')

import time
import math
import models
import torch
import utils
from tqdm import trange
from routing import *
from dictionary import DCTDictionary
from ksvd import KSVD
from pursuit import MatchingPursuit, Solver_l0
import pickle
import warnings

# ssh aiotlab@202.191.57.61 -p 1111

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def get_psi(args, samples=4000, iterator=100):
    X = utils.load_raw(args)

    X = X[:samples]

    X_temp = np.array([np.max(X[args.seq_len_x + i:
                                args.seq_len_x + i + args.seq_len_y], axis=0) for i in
                       range(samples - args.seq_len_x - args.seq_len_y)]).T

    size_D = int(math.sqrt(X.shape[1]))

    D = DCTDictionary(size_D, size_D)

    psi, _ = KSVD(D, MatchingPursuit, int(args.mon_rate / 100 * X.shape[1])).fit(X_temp, iterator)
    return psi


def get_phi(top_k_index, nseries):
    G = np.zeros((top_k_index.shape[0], nseries))

    for i, j in enumerate(G):
        j[top_k_index[i]] = 1

    return G


def main(args, **model_kwargs):
    device = torch.device(args.device)
    args.device = device
    if 'abilene' in args.dataset:
        args.nNodes = 12
        args.day_size = 288
    elif 'geant' in args.dataset:
        args.nNodes = 22
        args.day_size = 96
    elif 'brain' in args.dataset:
        args.nNodes = 9
        args.day_size = 1440
    elif 'sinet' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    elif 'renater' in args.dataset:
        args.nNodes = 30
        args.day_size = 288
    elif 'surfnet' in args.dataset:
        args.nNodes = 50
        args.day_size = 288
    elif 'uninett' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    else:
        raise ValueError('Dataset not found!')

    sets = ['train']
    for set in sets:
        if set == 'train' or set == 'val':
            args.test = False
            args.testset = 0
        else:
            args.test = True
            testset = int(set.split('_')[1])
            args.testset = testset

        train_loader, val_loader, test_loader, total_timesteps, total_series = utils.get_dataloader(args)
        args.nSeries = int(args.mon_rate * total_series / 100)

        in_dim = 1
        if args.tod:
            in_dim += 1
        if args.ma:
            in_dim += 1
        if args.mx:
            in_dim += 1

        args.in_dim = in_dim

        aptinit, supports = utils.make_graph_inputs(args, device)

        model = models.GWNet.from_args(args, supports, aptinit, **model_kwargs)
        model.to(device)
        logger = utils.Logger(args)

        engine = utils.Trainer.from_args(model=model, scaler=None,
                                         scaler_top_k=test_loader.dataset.scaler_topk, args=args)

        utils.print_args(args)
        if set == 'train':
            data_loader = train_loader
        elif set == 'val':
            data_loader = val_loader
        else:
            data_loader = test_loader

        # Metrics on test data
        engine.model.load_state_dict(torch.load(logger.best_model_save_path))
        with torch.no_grad():
            test_met_df, x_gt, y_gt, yhat, y_real = engine.test(data_loader, engine.model, args.out_seq_len)

        x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
        y_gt = y_gt.cpu().data.numpy()
        yhat = yhat.cpu().data.numpy()
        top_k_index = test_loader.dataset.Topkindex
        ygt_shape = y_gt.shape
        if args.cs:
            print('|--- Traffic reconstruction using CS')
            y_cs = np.zeros(shape=(ygt_shape[0], 1, ygt_shape[-1]))

            # obtain psi, G, R
            psi_save_path = os.path.join(args.datapath, 'cs/saved_psi/')
            if not os.path.exists(psi_save_path):
                os.makedirs(psi_save_path)
            psi_save_path = os.path.join(psi_save_path, '{}_{}_{}_{}_psi.pkl'.format(args.dataset,
                                                                                     args.mon_rate,
                                                                                     args.seq_len_x,
                                                                                     args.seq_len_y))
            if not os.path.isfile(psi_save_path):
                print('|--- Calculating psi, phi')

                psi = get_psi(args)
                obj = {
                    'psi': psi,
                }
                with open(psi_save_path, 'wb') as fp:
                    pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    fp.close()
            else:
                print('|--- Loading psi, phi from {}'.format(psi_save_path))

                with open(psi_save_path, 'rb') as fp:
                    obj = pickle.load(fp)
                    fp.close()
                psi = obj['psi']

            phi = get_phi(top_k_index, total_series)

            # traffic reconstruction using compressive sensing
            A = np.dot(phi, psi.matrix)
            for i in range(y_cs.shape[0]):
                sparse = Solver_l0(A, max_iter=100, sparsity=int(args.mon_rate / 100 * y_cs.shape[-1])).fit(yhat[i].T)
                y_cs[i] = np.dot(psi.matrix, sparse).T

        else:
            print('|--- No traffic reconstruction')
            y_cs = np.ones(shape=(ygt_shape[0], 1, ygt_shape[-1]))
            # y_cs[:] = np.mean(yhat)
            y_cs[:, :, top_k_index] = yhat

        x_gt = torch.from_numpy(x_gt).to(args.device)
        y_gt = torch.from_numpy(y_gt).to(args.device)
        y_cs = torch.from_numpy(y_cs).to(args.device)
        y_cs[y_cs < 0.0] = 0.0

        # run traffic engineering
        x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
        y_gt = y_gt.cpu().data.numpy()
        y_cs = y_cs.cpu().data.numpy()
        y_real = y_real.cpu().data.numpy()

        np.save(os.path.join(logger.log_dir, 'x_gt_test_{}'.format(args.testset)), x_gt)
        np.save(os.path.join(logger.log_dir, 'y_gt_test_{}'.format(args.testset)), y_gt)
        np.save(os.path.join(logger.log_dir, 'y_cs_test_{}'.format(args.testset)), y_cs)
        np.save(os.path.join(logger.log_dir, 'y_real_test_{}'.format(args.testset)), y_real)

        if args.run_te != 'None':
            if args.verbose:
                print('x_gt ', x_gt.shape)
                print('y_gt ', y_gt.shape)
                print('y_cs ', y_cs.shape)

            args.testset = set
            run_te(x_gt, y_gt, y_cs, args)

        print(
            '\n{} testset: {} x: {} y: {} topk:{} cs: {}'.format(args.dataset, args.testset, args.seq_len_x,
                                                                 args.seq_len_y,
                                                                 args.mon_rate, args.cs))
        print('\n            ----------------------------\n')


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
