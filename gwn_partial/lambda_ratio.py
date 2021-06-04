import sys

sys.path.append('..')

import time
import torch
import utils
from routing import *

import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def main(args, **model_kwargs):
    device = torch.device(args.device)
    args.device = device
    if args.dataset == 'abilene_tm':
        args.nNodes = 12
        args.day_size = 288
    elif args.dataset == 'geant_tm':
        args.nNodes = 22
        args.day_size = 96
    elif args.dataset == 'brain_tm':
        args.nNodes = 9
        args.day_size = 1440
    elif 'sinet' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    else:
        raise ValueError('Dataset not found!')

    train_loader, val_loader, test_loader, graphs = utils.get_dataloader(args)

    X = test_loader.dataset.X.cpu().numpy()
    lamda_ratios = []
    for i in range(0, X.shape[0], args.seq_len_y):

        max_max = np.max(X[i:i + args.seq_len_y])
        sum_max = np.sum(np.max(X[i:i + args.seq_len_y], axis=0))
        if max_max != 0:
            _r = sum_max / max_max
        else:
            _r = 0
        lamda_ratios.append(_r)
        print(i, ': ', _r)

    lamda_ratios = np.array(lamda_ratios)


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
