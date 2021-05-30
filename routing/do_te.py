from prophet import Prophet
from tqdm import tqdm

from .ls2sr import LS2SRSolver
from .max_step_sr import MaxStepSRSolver
from .multi_step_sr import MultiStepSRSolver
from .oblivious_routing import ObliviousRoutingSolver
from .one_step_sr import OneStepSRSolver
from .util import *


def calculate_lamda(y_gt):
    sum_max = np.sum(np.max(y_gt, axis=1))
    maxmax = np.max(y_gt)
    return sum_max / maxmax


def get_route_changes(routings, graph):
    route_changes = np.zeros(shape=(routings.shape[0] - 1))
    for t in range(routings.shape[0] - 1):
        _route_changes = 0
        for i, j in itertools.product(range(routings.shape[1]), range(routings.shape[2])):
            path_t_1 = get_paths_from_sulution(graph, routings[t + 1], i, j)
            path_t = get_paths_from_sulution(graph, routings[t], i, j)
            if path_t_1 != path_t:
                _route_changes += 1

        route_changes[t] = _route_changes

    return route_changes


def get_route_changes_heuristic(routings):
    route_changes = []
    for t in range(routings.shape[0] - 1):
        route_changes.append(count_routing_change(routings[t + 1], routings[t]))

    route_changes = np.asarray(route_changes)
    return route_changes


def extract_results(results):
    mlus, solutions = [], []
    for _mlu, _solution in results:
        mlus.append(_mlu)
        solutions.append(_solution)

    mlus = np.stack(mlus, axis=0)
    solutions = np.stack(solutions, axis=0)

    return mlus, solutions


def save_results(log_dir, fname, mlus, route_change):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    np.save(os.path.join(log_dir, fname + '_mlus'), mlus)
    np.save(os.path.join(log_dir, fname + '_route_change'), route_change)


def prepare_te_data(x_gt, y_gt, yhat, args):
    te_step = args.test_size if args.te_step == 0 else args.te_step
    x_gt = x_gt[0:te_step:args.seq_len_y]
    y_gt = y_gt[0:te_step:args.seq_len_y]
    if args.run_te == 'ls2sr' or args.run_te == 'onestep':
        yhat = yhat[0:te_step:args.seq_len_y]

    return x_gt, y_gt, yhat


def oblivious_routing_solver(y_gt, G, segments, te_step, args):
    solver = ObliviousRoutingSolver(G, segments)
    solver.solve()
    print('Solving Obilious Routing: Done')
    results = []

    def f(tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        tms[tms <= 0.0] = 0.0
        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        return oblivious_sr(solver, tms)

    for i in tqdm(range(te_step)):
        results.append(f(tms=y_gt[i]))

    mlu, solutions = extract_results(results)
    rc = get_route_changes(solutions, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))

    print('Oblivious              | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                              np.mean(mlu),
                                                                              np.max(mlu),
                                                                              np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'oblivious', mlu, rc)


def last_step_solver(y_gt, x_gt, graph, segments, args):
    solver = OneStepSRSolver(graph, segments)

    def f(gt_tms, last_tm):
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms[gt_tms <= 0.0] = 0.0
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        last_tm[last_tm <= 0.0] = 0.0
        last_tm = last_tm.reshape((args.nNodes, args.nNodes))
        last_tm = last_tm * (1.0 - np.eye(args.nNodes))

        return last_step_sr(solver, last_tm, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(gt_tms=y_gt[i], last_tm=x_gt[i, -1, ...])
                                                  for i in range(x_gt.shape[0]))

    mlu, solution = extract_results(results)
    rc = get_route_changes(solution, graph)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc),
                                                        np.std(rc)))
    print('last-step            | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                            np.mean(mlu),
                                                                            np.max(mlu),
                                                                            np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'laststep', mlu, rc)


def first_step_solver(y_gt, G, segments, te_step, args):
    solver = OneStepSRSolver(G, segments)

    def f(gt_tms, first_tm):
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms[gt_tms <= 0.0] = 0.0
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        first_tm[first_tm <= 0.0] = 0.0
        first_tm = first_tm.reshape((args.nNodes, args.nNodes))
        first_tm = first_tm * (1.0 - np.eye(args.nNodes))

        return first_step_sr(solver, first_tm, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(gt_tms=y_gt[i], first_tm=y_gt[i, 0, ...])
                                                  for i in range(te_step))

    mlu, solution = extract_results(results)
    rc = get_route_changes(solution, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc),
                                                        np.std(rc)))
    print('first-step            | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                             np.mean(mlu),
                                                                             np.max(mlu),
                                                                             np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'first_step', mlu, rc)


def one_step_predicted_solver(yhat, y_gt, G, segments, te_step, args):
    solver = OneStepSRSolver(G, segments)

    def f(gt_tms, tm):
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms[gt_tms <= 0.0] = 0.0
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        tm[tm <= 0.0] = 0.0
        tm = tm.reshape((args.nNodes, args.nNodes))
        tm = tm * (1.0 - np.eye(args.nNodes))

        return one_step_predicted_sr(solver=solver, tm=tm, gt_tms=gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(gt_tms=y_gt[i], tm=yhat[i]) for i in range(te_step))

    mlu, solutions = extract_results(results)
    rc = get_route_changes(solutions, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(rc),
                                                        np.std(rc)))
    print('Ones-step prediction    | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                               np.mean(mlu),
                                                                               np.max(mlu),
                                                                               np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'one_step_pred_heiristic_{}'.format(args.model), mlu, rc)


def ls2sr_p0(yhat, y_gt, x_gt, G, segments, te_step, args):
    print('ls2sr_p0')
    solver = LS2SRSolver(G, args=args)

    results = Parallel(n_jobs=os.cpu_count() - 8)(delayed(p0_ls2sr)(
        solver=solver, tms=y_gt[i], gt_tms=y_gt[i], p_solution=None, nNodes=args.nNodes)
                                                  for i in range(te_step))

    mlu, solution = extract_results(results)
    rc = get_route_changes_heuristic(solution)
    print(
        'Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc),
                                                      np.std(rc)))
    print('P0 Heuristic    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                               np.min(mlu),
                                                                               np.mean(mlu),
                                                                               np.max(mlu),
                                                                               np.std(mlu)))

    save_results(args.log_dir, 'p0_ls2sr', mlu, rc)


def ls2sr_gwn_p2(yhat, x_gt, y_gt, graph, te_step, args):
    print('ls2sr_gwn_p2')

    alpha = 0.8

    results = []
    solver = LS2SRSolver(graph=graph, args=args)

    solution = None
    dynamicity = np.zeros(shape=(te_step, 7))
    for i in range(te_step):
        mean = np.mean(y_gt[i], axis=1)
        std_mean = np.std(mean)
        std = np.std(y_gt[i], axis=1)
        std_std = np.std(std)

        maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

        theo_lamda = calculate_lamda(y_gt=y_gt[i])

        pred_tm = alpha * yhat[i, 0, :] + (1.0 - alpha) * x_gt[i, -1, :]
        # pred_tm = np.zeros_like(yhat[i])
        u, solution = p2_heuristic_solver(solver, tm=pred_tm,
                                          gt_tms=y_gt[i], p_solution=solution, nNodes=args.nNodes)
        dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

        _solution = np.copy(solution)
        results.append((u, _solution))

    mlu, solution = extract_results(results)
    route_changes = get_route_changes_heuristic(solution)

    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                        (args.seq_len_y * route_changes.shape[0]),
                                                        np.std(route_changes)))
    print('P2 ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                           np.min(mlu),
                                                                           np.mean(mlu),
                                                                           np.max(mlu),
                                                                           np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'ls2sr_p2', mlu, route_changes)
    np.save(os.path.join(args.log_dir, 'ls2sr_p2_dyn'), dynamicity)


def ls2sr_p2(y_gt, graph, te_step, args):
    print('ls2sr p2')

    results = []
    solver = LS2SRSolver(graph=graph, args=args)

    solution = None
    dynamicity = np.zeros(shape=(te_step, 7))
    for i in range(te_step):
        mean = np.mean(y_gt[i], axis=1)
        std_mean = np.std(mean)
        std = np.std(y_gt[i], axis=1)
        std_std = np.std(std)

        maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

        theo_lamda = calculate_lamda(y_gt=y_gt[i])

        pred_tm = np.max(y_gt[i], axis=0, keepdims=True)
        u, solution = p2_heuristic_solver(solver, tm=pred_tm,
                                          gt_tms=y_gt[i], p_solution=solution, nNodes=args.nNodes)
        dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

        _solution = np.copy(solution)
        results.append((u, _solution))

    mlu, solution = extract_results(results)
    route_changes = get_route_changes_heuristic(solution)

    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                        (args.seq_len_y * route_changes.shape[0]),
                                                        np.std(route_changes)))
    print('P2 ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                           np.min(mlu),
                                                                           np.mean(mlu),
                                                                           np.max(mlu),
                                                                           np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'ls2sr_p2', mlu, route_changes)
    np.save(os.path.join(args.log_dir, 'ls2sr_p2_dyn'), dynamicity)


def last_step_ls2sr(y_gt, x_gt, graph, te_step, args):
    print('last_step_ls2sr solver')

    results = []
    solver = LS2SRSolver(graph=graph, args=args)

    solution = None
    dynamicity = np.zeros(shape=(te_step, 7))
    for i in range(te_step):
        mean = np.mean(y_gt[i], axis=1)
        std_mean = np.std(mean)
        std = np.std(y_gt[i], axis=1)
        std_std = np.std(std)

        maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

        theo_lamda = calculate_lamda(y_gt=y_gt[i])

        last_tm = x_gt[i, -1, :]
        u, solution = p2_heuristic_solver(solver, tm=last_tm,
                                          gt_tms=y_gt[i], p_solution=solution, nNodes=args.nNodes)
        dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

        _solution = np.copy(solution)
        results.append((u, _solution))

    mlu, solution = extract_results(results)
    route_changes = get_route_changes_heuristic(solution)

    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                        (args.seq_len_y * route_changes.shape[0]),
                                                        np.std(route_changes)))
    print('last_step ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                                  np.min(mlu),
                                                                                  np.mean(mlu),
                                                                                  np.max(mlu),
                                                                                  np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'ls2sr_last_step', mlu, route_changes)
    np.save(os.path.join(args.log_dir, 'ls2sr_last_step_dyn'), dynamicity)


def prophet_predicted_solver(x_gt, y_gt, graph, te_step, args):
    print('ls2sr solver')

    prophet = Prophet()

    def prophet_prediction(input):
        prophet.fit(input)

    alpha = 0.7

    results = []
    solver = LS2SRSolver(graph=graph, args=args)

    solution = None
    dynamicity = np.zeros(shape=(te_step, 7))
    for i in range(te_step):
        mean = np.mean(y_gt[i], axis=1)
        std_mean = np.std(mean)
        std = np.std(y_gt[i], axis=1)
        std_std = np.std(std)

        maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

        theo_lamda = calculate_lamda(y_gt=y_gt[i])

        pred_tm = alpha * yhat[i] + (1.0 - alpha) * x_gt[i, -1, :]
        u, solution = p2_heuristic_solver(solver, tm=pred_tm,
                                          gt_tms=y_gt[i], p_solution=solution, nNodes=args.nNodes)
        dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

        _solution = np.copy(solution)
        results.append((u, _solution))

    mlu, solution = extract_results(results)
    route_changes = get_route_changes_heuristic(solution)

    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                        (args.seq_len_y * route_changes.shape[0]),
                                                        np.std(route_changes)))
    print('P2 ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                           np.min(mlu),
                                                                           np.mean(mlu),
                                                                           np.max(mlu),
                                                                           np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'ls2sr_p2', mlu, route_changes)
    np.save(os.path.join(args.log_dir, 'ls2sr_p2_dyn'), dynamicity)


def optimal_p0_solver(y_gt, G, segments, te_step, args):
    solver = OneStepSRSolver(G, segments)

    def f(gt_tms):
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms[gt_tms <= 0.0] = 0.0
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        return p0(solver, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution = extract_results(results)
    solution = np.reshape(solution, newshape=(-1, args.nNodes, args.nNodes, args.nNodes))
    rc = get_route_changes(solution, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal              | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                            np.mean(mlu),
                                                                            np.max(mlu),
                                                                            np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p0_optimal', mlu, rc)


def optimal_p1_solver(y_gt, G, segments, te_step, args):
    solver = MultiStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))
        return p1(solver, tms, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        tms=y_gt[i], gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution = extract_results(results)
    rc = get_route_changes(solution, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('P1                   | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                            np.mean(mlu),
                                                                            np.max(mlu),
                                                                            np.std(mlu)))

    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p1_optimal', mlu, rc)


def optimal_p2_solver(y_gt, G, segments, te_step, args):
    solver = MaxStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))
        tms = tms.reshape((args.nNodes, args.nNodes))

        return p2(solver, tms=tms, gt_tms=gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        gt_tms=y_gt[i], tms=np.max(y_gt[i], axis=0, keepdims=True)) for i in range(te_step))

    mlu, solution_optimal_p2 = extract_results(results)
    rc = get_route_changes(solution_optimal_p2, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('P2                   | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                            np.mean(mlu),
                                                                            np.max(mlu),
                                                                            np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p2_optimal', mlu, rc)


def optimal_p3_solver(y_gt, G, segments, te_step, args):
    t_prime = int(args.seq_len_y / args.trunk)
    solver = MultiStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        return p3(solver, tms, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        tms=np.stack([np.max(y_gt[i][j:j + t_prime], axis=0) for j in range(0, y_gt[i].shape[0], t_prime)]),
        gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution_optimal_p3 = extract_results(results)
    rc = get_route_changes(solution_optimal_p3, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('P3                   | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                            np.mean(mlu),
                                                                            np.max(mlu),
                                                                            np.std(mlu)))
    congested = mlu[mlu > 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p3_optimal', mlu, rc)


def p1(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p3(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p2(solver, tms, gt_tms):
    u = []

    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p0_ls2sr(solver, tms, gt_tms, p_solution, nNodes):
    u = []
    tms = tms.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tms = tms.reshape((-1, nNodes, nNodes))

    solutions = []
    for i in range(gt_tms.shape[0]):
        solution = solver.solve(tms[i], solution=p_solution)  # solve backtrack solution (line 131)
        u.append(solver.evaluate(solution, gt_tms[i]))
        solutions.append(solution)

    return u, solutions


def p2_heuristic_solver(solver, tm, gt_tms, p_solution, nNodes):
    u = []
    tm = tm.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tm[tm <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tm[:] = tm[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tm = tm.reshape((nNodes, nNodes))

    try:
        solution = solver.solve(tm, solution=p_solution)  # solve backtrack solution (line 131)
    except:
        solution = solver.initialize()

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(solution, gt_tms[i]))
    return u, solution


def last_step_sr(solver, last_tm, gt_tms):
    u = []
    try:
        solver.solve(last_tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def first_step_sr(solver, first_tm, gt_tms):
    u = []
    try:
        solver.solve(first_tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def one_step_predicted_sr(solver, tm, gt_tms):
    u = []
    try:
        solver.solve(tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p0(solver, gt_tms):
    u = []
    solutions = []
    for i in range(gt_tms.shape[0]):
        try:
            solver.solve(gt_tms[i])
        except:
            pass
        solutions.append(solver.solution)
        u.append(solver.evaluate(gt_tms[i]))

    solutions = np.stack(solutions, axis=0)
    return u, solutions


def oblivious_sr(solver, tms):
    u = []
    for i in range(tms.shape[0]):
        u.append(solver.evaluate(tms[i]))

    return u, solver.solution


def run_te(x_gt, y_gt, yhat, args):
    print('|--- run TE on DIRECTED graph')
    graph = load_network_topology(args.dataset, args.datapath)

    x_gt, y_gt, yhat = prepare_te_data(x_gt, y_gt, yhat, args)

    te_step = x_gt.shape[0]
    print('    Method           |   Min     Avg    Max     std')

    if args.run_te == 'ls2sr':
        ls2sr_gwn_p2(yhat, x_gt, y_gt, graph, te_step, args)
    elif args.run_te == 'ls2sr_p2':
        ls2sr_p2(y_gt, graph, te_step, args)
    elif args.run_te == 'p0':
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p0_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'p1':
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p1_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'p2':
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p2_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'p3':
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p3_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'onestep':
        segments = compute_path(graph, args.dataset, args.datapath)
        one_step_predicted_solver(yhat, y_gt, graph, segments, te_step, args)
    elif args.run_te == 'prophet':
        prophet_predicted_solver(x_gt, y_gt, graph, te_step, args)
    elif args.run_te == 'laststep':
        segments = compute_path(graph, args.dataset, args.datapath)
        last_step_solver(y_gt, x_gt, graph, segments, args)
    elif args.run_te == 'laststep_ls2sr':
        last_step_ls2sr(y_gt, x_gt, graph, te_step, args)
    elif args.run_te == 'firststep':
        segments = compute_path(graph, args.dataset, args.datapath)
        first_step_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'or':
        segments = compute_path(graph, args.dataset, args.datapath)
        oblivious_routing_solver(y_gt, graph, segments, te_step, args)
    else:
        raise RuntimeError('TE not found!')
