import random as rd
import time

import numpy as np

import utils


def main(args, **kwargs):
    X = utils.load_raw(args)
    train_size = int(X.shape[0] * 0.7)
    if (args.cs):
        random_time_step = rd.randint(0, train_size)
        top_k = int(args.mon_rate / 100 * X.shape[1])
        if (top_k < 1): top_k = 1
        top_k_index = utils.largest_indices(X[random_time_step], top_k)
        top_k_index = np.sort(top_k_index)[0]

        if (args.top_k_random):
            top_k_index = np.random.randint(X.shape[1], size=top_k_index.shape[0])

    X = X[:train_size, top_k_index]
    dataset = args.dataset.split('_')[0]

    print('Data Shape: {}'.format(X.shape))
    print("Time period you want to know ? h='Hour', d='Day', m='Month'")
    time_period = input()

    if (dataset == 'abilene'):
        if (time_period == 'h'):
            data = np.array([np.mean(X[i: i + 12], axis=0) for i in
                             range(0, X.shape[0] - X.shape[0] % 12, 12)])
        elif (time_period == 'd'):
            data = np.array([np.mean(X[i: i + 288], axis=0) for i in
                             range(0, X.shape[0] - X.shape[0] % 288, 288)])
        elif (time_period == 'm'):
            data = np.array([np.mean(X[i: i + 8640], axis=0) for i in
                             range(0, X.shape[0] - X.shape[0] % 8640, 8640)])
    elif (dataset == 'geant'):
        if (time_period == 'h'):
            data = np.array([np.mean(X[i: i + 4], axis=0) for i in
                             range(0, X.shape[0] - X.shape[0] % 4, 4)])
        elif (time_period == 'd'):
            data = np.array([np.mean(X[i: i + 96], axis=0) for i in
                             range(0, X.shape[0] - X.shape[0] % 96, 96)])
        elif (time_period == 'm'):
            data = np.array([np.mean(X[i: i + 2880], axis=0) for i in
                             range(0, X.shape[0] - X.shape[0] % 2880, 2880)])
    elif (dataset == 'brain'):
        if (time_period == 'h'):
            data = np.array([np.mean(X[i: i + 60], axis=0) for i in
                             range(0, X.shape[0] - X.shape[0] % 60, 60)])
        elif (time_period == 'd'):
            data = np.array([np.mean(X[i: i + 1440], axis=0) for i in
                             range(0, X.shape[0] - X.shape[0] % 1440, 1440)])
        elif (time_period == 'm'):
            print('Brain data was measured under 7 days!!')
    # data = data.reshape(-1, 1)
    print('MAX: {}, MIN: {}'.format(np.max(data), np.min(data)))

    # if not os.path.exists(os.path.join(os.getcwd(), 'plot_traffic')): os.mkdir('plot_traffic')
    # plt.hist(data, bins=12, color='green', range=(np.min(data), np.max(data)), log=True, density=1)
    # plt.title(dataset + '_' + time_period)
    # plt.savefig('plot_traffic/' + dataset + '_' + time_period + '.png')

    var = np.var(data, axis=0)
    if (time_period == 'h'):
        print('Variance of {} by Hour: {}'.format(dataset, var))
    elif (time_period == 'd'):
        print('Variance of {} by Day: {}'.format(dataset, var))
    elif (time_period == 'm'):
        print('Variance of {} by Month: {}'.format(dataset, var))


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
