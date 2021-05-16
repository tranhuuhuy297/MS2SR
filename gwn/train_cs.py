import sys

sys.path.append('..')

import time
import math
import models
import torch
import utils
from tqdm import trange
from routing import *
from utils import *
from dictionary import DCTDictionary
from ksvd import KSVD
from pursuit import MatchingPursuit, Solver_l0
import pickle
import warnings

# ssh aiotlab@202.191.57.61 -p 1111

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def get_psi(args, samples=10000, iterator=100):
    X = utils.load_raw(args)

    if (X.shape[0] < 10000): samples = 4000

    X = X[:samples, :]

    X_temp = np.array([np.max(X[args.seq_len_x + i: \
                                args.seq_len_x + i + args.seq_len_y], axis=0) for i in
                       range(samples - args.seq_len_x - args.seq_len_y)]).T

    size_D = int(math.sqrt(X.shape[1]))

    D = DCTDictionary(size_D, size_D)

    psi, _ = KSVD(D, MatchingPursuit, int(args.random_rate / 100 * X.shape[1])).fit(X_temp, iterator)
    return psi


def get_phi(args, top_k_index):
    X = utils.load_raw(args)
    G = np.zeros((top_k_index.shape[0], X.shape[1]))

    for i, j in enumerate(G):
        j[top_k_index[i]] = 1

    return G


def main(args, **model_kwargs):
    # # psi
    # psi = get_psi(args).matrix

    # # phi
    # train_loader, val_loader, test_loader, top_k_index = utils.get_dataloader(args)
    # phi = get_phi(args, top_k_index)

    aptinit, supports = utils.make_graph_inputs(args, args.device)
    model = models.GWNet.from_args(args, supports, aptinit, **model_kwargs)
    model.to(args.device)
    logger = utils.Logger(args)
    engine = utils.Trainer.from_args(model, train_loader.dataset.scaler, train_loader.dataset.scaler_top_k, args)
    engine.model.load_state_dict(torch.load(logger.best_model_save_path))
    with torch.no_grad():
        test_met_df, x_gt, y_gt, y_real, yhat = engine.test(test_loader, engine.model, args.out_seq_len)

    x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
    y_gt = y_gt.cpu().data.numpy()
    yhat = yhat.cpu().data.numpy()

    ygt_shape = y_gt.shape
    y_cs = np.zeros(shape=(ygt_shape[0], 1, ygt_shape[-1]))

    print(yhat.shape, y_gt.shape)

if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
