import subprocess as sp

from tqdm import trange


def call(args):
    p = sp.run(args=args,
               stdout=sp.PIPE,
               stderr=sp.PIPE)
    stdout = p.stdout.decode('utf-8')
    return stdout


def main():
    # get args
    dataset_name = 'geant'
    random_rate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    CS = [0, 1]
    testset = [0, 1, 2, 3, 4, 5]

    iteration = trange(len(random_rate))
    # experiment for each dataset
    for d in iteration:
        for test in testset:
            for cs in CS:
                args = ['python',
                        'train.py',
                        '--do_graph_conv --aptonly --addaptadj --randomadj',
                        '--train_batch_size 64 --val_batch_size 64',
                        '--test --run_te gwn_ls2sr',
                        '--device', 'cuda:0',
                        '--random_rate', str(random_rate[d]),
                        '--testset', str(test),
                        '--cs', str(cs)]
                stdout = call(args)

                iteration.set_description(
                    'Dataset {} random_rate: {} - testset {} - cs {}'.format(dataset_name, random_rate[d], test, cs))


if __name__ == '__main__':
    main()
