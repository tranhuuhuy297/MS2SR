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
    else:
        raise ValueError('Dataset not found!')

    train_loader, val_loader, test_loader, top_k_index = utils.get_dataloader(args)

    args.train_size, args.nSeries = train_loader.dataset.X_scaled_top_k.shape
    args.val_size = val_loader.dataset.X_scaled_top_k.shape[0]
    args.test_size = test_loader.dataset.X_scaled_top_k.shape[0]

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

    engine = utils.Trainer.from_args(model, train_loader.dataset.scaler, train_loader.dataset.scaler_top_k, args)

    utils.print_args(args)

    if not args.test:
        iterator = trange(args.epochs)

        try:
            if os.path.isfile(logger.best_model_save_path):
                print('Model checkpoint exist!')
                print('Load model checkpoint? (y/n)')
                _in = input()
                if _in == 'y' or _in == 'yes':
                    print('Loading model...')
                    engine.model.load_state_dict(torch.load(logger.best_model_save_path))
                else:
                    print('Training new model')

            for epoch in iterator:
                train_loss, train_rse, train_mae, train_mse, train_mape, train_rmse = [], [], [], [], [], []
                for iter, batch in enumerate(train_loader):

                    # x = batch['x']  # [b, seq_x, n, f]
                    # y = batch['y']  # [b, seq_y, n]
                    # sys.exit()
                    x = batch['x_top_k']
                    y = batch['y_top_k']

                    if y.max() == 0: continue
                    loss, rse, mae, mse, mape, rmse = engine.train(x, y)
                    train_loss.append(loss)
                    train_rse.append(rse)
                    train_mae.append(mae)
                    train_mse.append(mse)
                    train_mape.append(mape)
                    train_rmse.append(rmse)

                engine.scheduler.step()
                with torch.no_grad():
                    val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = engine.eval(val_loader)
                m = dict(train_loss=np.mean(train_loss), train_rse=np.mean(train_rse),
                         train_mae=np.mean(train_mae), train_mse=np.mean(train_mse),
                         train_mape=np.mean(train_mape), train_rmse=np.mean(train_rmse),
                         val_loss=np.mean(val_loss), val_rse=np.mean(val_rse),
                         val_mae=np.mean(val_mae), val_mse=np.mean(val_mse),
                         val_mape=np.mean(val_mape), val_rmse=np.mean(val_rmse))

                description = logger.summary(m, engine.model)

                if logger.stop:
                    break

                description = 'Epoch: {} '.format(epoch) + description
                iterator.set_description(description)
        except KeyboardInterrupt:
            pass

    # Metrics on test data
    engine.model.load_state_dict(torch.load(logger.best_model_save_path))
    with torch.no_grad():
        test_met_df, x_gt, y_gt, y_real, yhat = engine.test(test_loader, engine.model, args.out_seq_len)

    x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
    y_gt = y_gt.cpu().data.numpy()
    yhat = yhat.cpu().data.numpy()

    ygt_shape = y_gt.shape
    y_cs = np.zeros(shape=(ygt_shape[0], 1, ygt_shape[-1]))

    y_temp = np.max(y_gt, axis=1)
    y_test_gt = y_temp.reshape(y_temp.shape[0], 1, y_temp.shape[-1])
    y_test = y_temp[:, top_k_index].reshape(y_temp.shape[0], 1, len(top_k_index))
    yhat = y_test

    psi = get_psi(args)
    phi = get_phi(args, top_k_index)

    A = np.dot(phi, psi.matrix)
    for i in range(y_cs.shape[0]):
        sparse = Solver_l0(A, max_iter=100, sparsity=int(args.random_rate / 100 * y_cs.shape[-1])).fit(yhat[i].T)
        y_cs[i] = np.dot(psi.matrix, sparse).T

    x_gt = torch.from_numpy(x_gt).to(args.device)
    y_gt = torch.from_numpy(y_gt).to(args.device)
    y_cs = torch.from_numpy(y_cs).to(args.device)
    y_cs[y_cs < 0] = 0
    test_met = []
    for i in range(y_cs.shape[1]):
        pred = y_cs[:, i, :]
        pred = torch.clamp(pred, min=0., max=10e10)
        real = y_real[:, i, :]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
    test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'test_metrics.csv'))
    print('Prediction Accuracy:')
    print(utils.summary(logger.log_dir))

if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
