import numpy as np
import torch
from scipy.stats import sem


class evaluator_all():
    def __init__(self, args):
        # super().__init__()

        self.reps = args.reps
        self.scores_train = np.zeros((self.reps, 3))
        self.scores_test = np.zeros((self.reps, 3))

    def pre_evaluate_init(self, train, valid, test, contfeats, binfeats):
        (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
        (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
        (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

        # reorder features with binary first and continuous after
        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]
        # concatenate train and valid for training
        xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), \
                                 np.concatenate([ttr, tva], axis=0), \
                                 np.concatenate([ytr, yva], axis=0)

        self.talltr, self.tte = talltr, tte

        self.evaluator_train = Evaluator(yalltr, talltr,
                                         y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                         mu0=np.concatenate([mu0tr, mu0va], axis=0),
                                         mu1=np.concatenate([mu1tr, mu1va], axis=0))
        self.evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

        self.evaluator_train_t0 = Evaluator(yalltr[talltr[:, 0] == 0], talltr[talltr[:, 0] == 0],
                                            y_cf=np.concatenate([y_cftr, y_cfva], axis=0)[talltr[:, 0] == 0],
                                            mu0=np.concatenate([mu0tr, mu0va], axis=0)[talltr[:, 0] == 0],
                                            mu1=np.concatenate([mu1tr, mu1va], axis=0)[talltr[:, 0] == 0])
        self.evaluator_test_t0 = Evaluator(yte[tte[:, 0] == 0], tte[tte[:, 0] == 0], y_cf=y_cfte[tte[:, 0] == 0],
                                           mu0=mu0te[tte[:, 0] == 0], mu1=mu1te[tte[:, 0] == 0])

        self.evaluator_train_t1 = Evaluator(yalltr[talltr[:, 0] == 1], talltr[talltr[:, 0] == 1],
                                            y_cf=np.concatenate([y_cftr, y_cfva], axis=0)[talltr[:, 0] == 1],
                                            mu0=np.concatenate([mu0tr, mu0va], axis=0)[talltr[:, 0] == 1],
                                            mu1=np.concatenate([mu1tr, mu1va], axis=0)[talltr[:, 0] == 1])
        self.evaluator_test_t1 = Evaluator(yte[tte[:, 0] == 1], tte[tte[:, 0] == 1], y_cf=y_cfte[tte[:, 0] == 1],
                                           mu0=mu0te[tte[:, 0] == 1], mu1=mu1te[tte[:, 0] == 1])

    def evaluates(self, i, y0_train_pred, y1_train_pred, y0_test_pred, y1_test_pred):
        score_train = self.evaluator_train.calc_stats(y1_train_pred, y0_train_pred)
        self.scores_train[i, :] = score_train

        score_test = self.evaluator_test.calc_stats(y1_test_pred, y0_test_pred)
        self.scores_test[i, :] = score_test
        print('Replication: {}/{}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}' \
              ', te_ate: {:0.3f}, te_pehe: {:0.3f}'.
              format(i + 1, self.reps, score_train[0], score_train[1], score_train[2],
                     score_test[0], score_test[1], score_test[2]))

    def cal_stats_train(self, y1_train_pred, y0_train_pred):
        return self.evaluator_train.calc_stats(y1_train_pred, y0_train_pred)

    def cal_stats_test(self, y1_test_pred, y0_test_pred):
        return self.evaluator_test.calc_stats(y1_test_pred, y0_test_pred)

    def format_result(self, means_tr, stds_tr, means_te, stds_te):
        msg_dict = {"trATE": "%.3f+-%.3f" % (means_tr[0], stds_tr[0]),
                    "trPEHE": "%.3f+-%.3f" % (means_tr[1], stds_tr[1]),
                    "trMSE": "%.3f+-%.3f" % (means_tr[2], stds_tr[2]),

                    "teATE": "%.3f+-%.3f" % (means_te[0], stds_te[0]),
                    "tePEHE": "%.3f+-%.3f" % (means_te[1], stds_te[1]),
                    "teMSE": "%.3f+-%.3f" % (means_te[2], stds_te[2]),
                    }

        return msg_dict

    def get_scores(self):
        print('\nResults')
        means_tr, stds_tr = np.mean(self.scores_train, axis=0), sem(self.scores_train, axis=0)
        msg = '  {:>10s} ATE: {:.3f}+-{:.3f}, PEHE: {:.3f}+-{:.3f}, MSE: {:.3f}+-{:.3f}'
        print(msg.format("train",
                         means_tr[0], stds_tr[0],
                         means_tr[1], stds_tr[1],
                         means_tr[2], stds_tr[2]))

        means_te, stds_te = np.mean(self.scores_test, axis=0), sem(self.scores_test, axis=0)
        print(msg.format(" test",
                         means_te[0], stds_te[0],
                         means_te[1], stds_te[1],
                         means_te[2], stds_te[2]))
        return self.format_result(means_tr, stds_tr, means_te, stds_te)


class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf

        if mu0 is None:
            self.mu0 = (1 - t) * y + t * y_cf
            self.mu1 = t * y + (1 - t) * y_cf
        else:
            self.mu0 = mu0
            self.mu1 = mu1

        self.true_ite = self.mu1 - self.mu0

    def abs_ate(self, ypred1, ypred0):
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def mse(self, ypred1, ypred0):
        ypred = (1 - self.t) * ypred0 + self.t * ypred1
        return np.mean(np.square(ypred - self.y))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        mse = self.mse(ypred1, ypred0)
        return ate, pehe, mse


def get_y0_y1(
        p_zu_x_dist, p_y_tu_dist
        , x, t):
    x = x.to(torch.float32)
    p_z_x, p_u_x, att = p_zu_x_dist(x)

    y0 = p_y_tu_dist(torch.zeros(t.shape).cuda(), p_u_x).detach().cpu().numpy()
    y1 = p_y_tu_dist(torch.ones(t.shape).cuda(), p_u_x).detach().cpu().numpy()

    return y0, y1, att
