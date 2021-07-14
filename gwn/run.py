import argparse
import os
import subprocess as sp

from tqdm import trange


def call(args):
    p = sp.run(args=args,
               stdout=sp.PIPE,
               stderr=sp.PIPE)
    stdout = p.stdout.decode('utf-8')
    return stdout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm', 'brain5_tm', 'brain15_tm', 'abilene15_tm',
                                 'brain10_tm', 'abilene10_tm'],
                        help='Dataset, (default abilene_tm)')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--run_te', type=str, choices=['None', 'gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2',
                                                       'p3', 'onestep', 'prophet', 'laststep', 'laststep_ls2sr',
                                                       'firststep', 'or'],
                        default='None')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fs', help='Flow selection strategiy', type=str, choices=['train'],
                        default='train')
    parser.add_argument('--mon_rate', type=int, default=5, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()
    dataset_name = args.dataset
    # mon_rate = [1, 2, 3, 4, 5]
    if args.mon_rate == 0:
        mon_rate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        mon_rate = [args.mon_rate]

    if args.test:
        CS = [0, 1]
        if args.dataset == 'abilene_tm':
            testset = [0, 1, 2, 3, 4, 5]
        else:
            testset = [0, 1, 2, 3, 4]
    else:
        CS = [1]
        testset = [0]
    iteration = trange(len(mon_rate))
    # experiment for each dataset
    for d in iteration:
        for test in testset:
            for cs in CS:
                cmd = 'python train.py --do_graph_conv --aptonly --addaptadj --randomadj'
                cmd += ' --train_batch_size 64 --val_batch_size 64'
                cmd += ' --dataset {}'.format(dataset_name)
                cmd += ' --mon_rate {}'.format(mon_rate[d])
                cmd += ' --device {}'.format(args.device)
                cmd += ' --fs {}'.format(args.fs)
                if args.test:
                    cmd += ' --test'
                    cmd += ' --testset {} --cs {}'.format(test, cs)

                if args.run_te != 'None':
                    cmd += ' --run_te {}'.format(args.run_te)

                os.system(cmd)
                iteration.set_description(
                    'Dataset {} mon_rate: {} - testset {} - cs {}'.format(dataset_name, mon_rate[d], test, cs))


if __name__ == '__main__':
    main()
