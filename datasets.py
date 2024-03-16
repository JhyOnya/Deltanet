import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sp


class Mdata(object):
    def __init__(self, path_data="", replications=10):
        self.path_data = path_data
        self.replications = replications

        self.binfeats = []
        self.contfeats = []

    def generate(self, sample_size=2000, m_dim=50, z_dim=50, u_dim=50):
        x_dim = 500 - m_dim
        z = np.random.normal(0, 1, (sample_size, z_dim))
        u = np.random.normal(0, 1, (sample_size, u_dim))

        x_bin = np.random.binomial(1, 0.5, (sample_size, int(x_dim / 5)))
        x_con = np.random.normal(0, 1, (sample_size, x_dim - x_bin.shape[1]))
        x = np.concatenate((x_bin, x_con), axis=1)

        w_zm = np.random.uniform(0, 1, (z_dim, m_dim))
        w_um = np.random.uniform(0, 1, (u_dim, m_dim))
        m = np.matmul(z, w_zm) + np.matmul(u, w_um)
        m = m + np.random.normal(0, 1, (sample_size, m_dim))

        w_zt = np.random.uniform(0, 1, (z_dim, 1))
        w_xt = np.random.uniform(0, 1, (x_dim, 1))
        t_con = np.matmul(z, w_zt) + np.matmul(x, w_xt)
        t = np.random.binomial(1, 1 / (1 + np.exp(-(t_con - np.mean(t_con)))), t_con.shape)

        w_xy = np.random.uniform(0, 1, (x_dim, 1))
        w_uy = np.random.uniform(0, 1, (u_dim, 1))
        w_ty = np.random.normal(0, 1, (sample_size, 1)) + 2
        y__ = np.matmul(x, w_xy) + np.matmul(u, w_uy)
        y__ = y__ + np.random.normal(0, 1, (sample_size, 1))
        y_f = y__ + t * w_ty
        y_c = y__ + (1 - t) * w_ty
        mu0 = y__
        mu1 = y__ + 1 * w_ty

        return np.concatenate((x, m), axis=1), t, y_f, y_c, mu0, mu1

    def get_train_valid_test(self):
        for i in range(self.replications):
            np.random.seed(i)

            x, t, y_f, y_cf, mu_0, mu_1 = self.generate()

            if len(self.binfeats) == 0 and len(self.contfeats) == 0 or True:
                self.binfeats = []
                self.contfeats = []
                x_pd = pd.DataFrame(x)
                for i in range(x_pd.shape[1]):
                    col_count = len(dict(x_pd.iloc[:, i].value_counts()))
                    if col_count == 2:
                        self.binfeats.append(i)
                        key_ls = sorted(dict(x_pd.iloc[:, i].value_counts()).keys())

                        if 0 in key_ls and 1 in key_ls:
                            pass
                        else:
                            x_pd.iloc[:, i] = x_pd.iloc[:, i].replace(key_ls[0], "0_")
                            x_pd.iloc[:, i] = x_pd.iloc[:, i].replace(key_ls[1], "1_")
                            x_pd.iloc[:, i] = x_pd.iloc[:, i].replace("0_", 0)
                            x_pd.iloc[:, i] = x_pd.iloc[:, i].replace("1_", 1)
                    else:
                        self.contfeats.append(i)

            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y_f[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y_f[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y_f[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats


class IHDP(object):
    def __init__(self, path_data="datasets/IHDP/", replications=10):
        self.path_data = path_data

        self.replications = replications

        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def get_train_valid_test(self):
        a_te = np.load(open(self.path_data + "ihdp_npci_1-100.test.npz", 'rb'), encoding='bytes')
        a_tr = np.load(open(self.path_data + "ihdp_npci_1-100.train.npz", 'rb'), encoding='bytes')

        for i in range(self.replications):
            t, y, y_cf = a_tr['t'][:, i].reshape(-1, 1), a_tr['yf'][:, i].reshape(-1, 1), a_tr['ycf'][:, i].reshape(-1,
                                                                                                                    1)
            mu_0, mu_1 = a_tr['mu0'][:, i].reshape(-1, 1), a_tr['mu1'][:, i].reshape(-1, 1)

            x = a_tr['x'][:, :, i]

            x[:, 13] -= 1  # this binary feature is in {1, 2}
            x_te = a_te['x'][:, :, i]
            x_te[:, 13] -= 1  # this binary feature is in {1, 2}

            itr, iva = train_test_split(np.arange(x.shape[0]), test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x_te, a_te['t'][:, i].reshape(-1, 1), a_te['yf'][:, i].reshape(-1, 1)), \
                   (
                       a_te['ycf'][:, i].reshape(-1, 1), a_te['mu0'][:, i].reshape(-1, 1),
                       a_te['mu1'][:, i].reshape(-1, 1))
            yield train, valid, test, self.contfeats, self.binfeats


class News(object):
    def __init__(self, path_data="datasets/News/csv/topic_doc_mean_n5000_k3477_seed_%d.csv.", replications=10):
        self.path_data = path_data
        self.replications = 50

        self.binfeats = []
        self.contfeats = []

    def get_train_valid_test(self):
        for i in range(self.replications):
            path_pre = self.path_data % (i + 1)

            a = np.loadtxt(path_pre + "x", delimiter=",")
            b = sp.coo_matrix(arg1=(a[1:, 2],
                                    (a[1:, 0].astype(np.int) - 1,
                                     a[1:, 1].astype(np.int) - 1)),
                              shape=(int(a[0, 0]), int(a[0, 1])),
                              dtype=np.float32)
            x = b.todense()

            if len(self.binfeats) == 0 and len(self.contfeats) == 0 or True:
                self.binfeats = []
                self.contfeats = []
                x_pd = pd.DataFrame(x)
                for i in range(x_pd.shape[1]):
                    col_count = len(dict(x_pd.iloc[:, i].value_counts()))
                    if col_count == 2:
                        self.binfeats.append(i)
                        key_ls = sorted(dict(x_pd.iloc[:, i].value_counts()).keys())

                        if 0 in key_ls and 1 in key_ls:
                            pass
                        else:
                            x_pd.iloc[:, i] = x_pd.iloc[:, i].replace(key_ls[0], "0_")
                            x_pd.iloc[:, i] = x_pd.iloc[:, i].replace(key_ls[1], "1_")
                            x_pd.iloc[:, i] = x_pd.iloc[:, i].replace("0_", 0)
                            x_pd.iloc[:, i] = x_pd.iloc[:, i].replace("1_", 1)
                    else:
                        self.contfeats.append(i)
                print("  binary", len(self.binfeats))
                print("  continuous", len(self.contfeats))

            y_all = np.loadtxt(path_pre + "y", delimiter=",")


            t = y_all[:, 0].reshape(-1, 1)
            y = y_all[:, 1].reshape(-1, 1)
            y_cf = y_all[:, 2].reshape(-1, 1)
            mu_0 = y_all[:, 3].reshape(-1, 1)
            mu_1 = y_all[:, 4].reshape(-1, 1)

            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats


if __name__ == "__main__":
    dataset = Mdata()
    for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
        print()
        (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
        (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
        (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
        # print("t=0", ttr[ttr == 0].shape)
        # print("t=1", ttr[ttr == 1].shape)
        # print("xtr", xtr.shape)
        # print("ttr", ttr.shape)
        # print("ytr", ytr.shape)
        # print("ycftr", y_cftr.shape)
        # print("mu0tr", mu0tr.shape)
        # print("mu1tr", mu1tr.shape)
