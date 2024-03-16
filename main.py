import numpy as np

import torch
from torch import optim

from datasets import Mdata, IHDP, News
from evaluation import get_y0_y1, evaluator_all
from networks import p_zu_x_func, p_t_z_func, p_y_tu_func

import reweighting as rw


# set random seeds:
def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_models(stat_params, cache_dir, i):
    stat_params_state_dict = {name: param_pre.state_dict() for name, param_pre in stat_params.items()}
    torch.save(stat_params_state_dict, "%sstats_%d.pth" % (cache_dir, i))


def restore_models(stat_params, cache_dir, i):
    checkpoint = torch.load("%sstats_%d.pth" % (cache_dir, i))
    for name, param_pre in stat_params.items():
        param_pre.load_state_dict(checkpoint[name])
        param_pre.eval()


def main(args):
    torch.cuda.set_device(0)
    # print(torch.cuda.is_available())
    setup_seed()

    dataset_dict = {"Mdata": Mdata, "IHDP": IHDP, "News": News}
    dataset = dataset_dict[args.data](replications=args.reps)

    eva = evaluator_all(args)

    l_mse = torch.nn.MSELoss(reduction='none')
    l_bce = torch.nn.BCELoss(reduction='none')

    # Loop for replications
    for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
        if i >= args.reps:
            break
        print('Replication %i/%i' % (i + 1, args.reps))

        eva.pre_evaluate_init(train, valid, test, contfeats, binfeats)

        (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
        (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
        (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

        xallva, tallva, yallva = np.concatenate([xtr, xva], axis=0), \
                                 np.concatenate([ttr, tva], axis=0), \
                                 np.concatenate([ytr, yva], axis=0)

        if args.weight_method == "nrw":
            weighttr = np.ones((xtr.shape[0], 1))
        else:
            weighttr = rw.decorrelation_pre(args.weight_method, xtr, ttr)

        best_logpvalid = np.inf

        p_zu_x_dist = p_zu_x_func(dim_in=len(binfeats) + len(contfeats), nh=2, dim_h=args.h_dim,
                                  dim_out=args.z_dim).cuda()
        p_t_z_dist = p_t_z_func(dim_in=args.z_dim, nh=2, dim_h=args.h_dim, dim_out=1).cuda()

        p_y_tu_dist = p_y_tu_func(dim_in=args.z_dim, nh=2, dim_h=args.h_dim, dim_out=1).cuda()

        # for save models
        stat_params = {  # 'att': p_att_x_dist,
            'p_zu_x': p_zu_x_dist,
            'p_t_z': p_t_z_dist,
            'p_y_tu': p_y_tu_dist}
        params = [pre for v in stat_params.values() for pre in list(v.parameters())]

        # Adam is used, like original implementation, in paper Adamax is suggested
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.decay)  # lr 学习率 wd 权重衰减

        n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / args.batch), list(range(xtr.shape[0]))

        for epoch in range(n_epoch):
            np.random.shuffle(idx)

            for j in range(n_iter_per_epoch):
                batch = np.random.choice(idx, args.batch)

                x_train, y_train, t_train = torch.cuda.FloatTensor(xtr[batch]), \
                                            torch.cuda.FloatTensor(ytr[batch]), \
                                            torch.cuda.FloatTensor(ttr[batch])

                p_z_x, p_u_x, att = p_zu_x_dist(x_train)

                p_t_z = p_t_z_dist(p_z_x)
                l_p_t_z = l_bce(p_t_z, t_train)

                p_y_tu = p_y_tu_dist(t_train, p_u_x)
                l_p_y_tu = l_mse(p_y_tu, y_train)

                w_train = torch.cuda.FloatTensor(weighttr[batch])

                if args.weight_method == "nrw":
                    loss_mean = torch.mean(l_p_t_z + l_p_y_tu)
                else:
                    loss_mean = torch.mean(w_train * ((l_p_t_z + l_p_y_tu).reshape(-1, 1)))

                objective = loss_mean

                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=3, norm_type=2)
                optimizer.step()

            y0_train_pred, y1_train_pred, att = get_y0_y1(p_zu_x_dist, p_y_tu_dist,
                                                          torch.tensor(xallva).cuda(),
                                                          torch.tensor(tallva).cuda())

            score_train = eva.cal_stats_train(y1_train_pred, y0_train_pred)
            logp_valid = score_train[1]

            if logp_valid <= best_logpvalid:
                best_logpvalid = logp_valid
                save_models(stat_params, args.cache_dir, i)

            if epoch % 10 == 0 or epoch == (n_epoch - 1):
                print('  Epoch %i' % (epoch + 1))
                print('    Train set - ate: %.3f, pehe: %.3f, mse: %.3f' % score_train)

                y0_test_pred, y1_test_pred, att = get_y0_y1(p_zu_x_dist, p_y_tu_dist,
                                                            torch.tensor(xte).cuda(),
                                                            torch.tensor(tte).cuda())
                score_test = eva.cal_stats_test(y1_test_pred, y0_test_pred)
                print('    Test  set - ate: %.3f, pehe: %.3f, mse: %.3f' % score_test)

        restore_models(stat_params, args.cache_dir, i)

        y0_train_pred, y1_train_pred, att = get_y0_y1(p_zu_x_dist, p_y_tu_dist,
                                                      torch.tensor(xallva).cuda(),
                                                      torch.tensor(tallva).cuda())

        y0_test_pred, y1_test_pred, att = get_y0_y1(p_zu_x_dist, p_y_tu_dist,
                                                    torch.tensor(xte).cuda(),
                                                    torch.tensor(tte).cuda())

        eva.evaluates(i, y0_train_pred, y1_train_pred, y0_test_pred, y1_test_pred)

    return eva.get_scores()
