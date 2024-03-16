import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


def decorrelation_mlp(x, t, solver='adam', hidden_layer_sizes=(100, 100), max_iter=1000, random_state=0):
    n, p = x.shape

    x_t1 = pd.DataFrame(x)[t == 1]
    x_t0 = pd.DataFrame(x)[t == 0]
    n_t1, _ = x_t1.shape
    n_t0, _ = x_t0.shape

    x_t1['src'] = 1  # 1 means source distribution
    x_t0['src'] = 0  # 0 means target distribution
    x_all = pd.concat([x_t1, x_t0], ignore_index=True, axis=0)
    labels = x_all['src'].values
    x_all = x_all.drop('src', axis=1).values
    x_t1, x_t0 = x_t1.values, x_t0.values

    # train a multi-layer perceptron to classify the source and target distribution
    clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
                        random_state=random_state)
    clf.fit(x_all, labels)

    proba1 = clf.predict_proba(x_all)[:len(x_t1), 1]
    proba0 = clf.predict_proba(x_all)[len(x_t1):, 0]

    proba1[proba1 == 0] = 1e-8
    proba0[proba0 == 0] = 1e-8

    weight1 = 1. / proba1
    weight0 = 1. / proba0

    weights = np.empty_like(t) * 1.
    weights[t == 1] = weight1
    weights[t == 0] = weight0

    weights = np.reshape(weights, [n, 1])
    return weights


def decorrelation_ed(x, t):
    n, p = x.shape

    x_t1 = pd.DataFrame(x)[t == 1].values
    x_t0 = pd.DataFrame(x)[t == 0].values
    n_t1, _ = x_t1.shape
    n_t0, _ = x_t0.shape

    x_t1_mean = x_t1.mean(axis=0)
    x_t0_mean = x_t0.mean(axis=0)

    proba1 = np.sqrt(np.sum(np.square((x_t1 - x_t1_mean)), axis=1)) / \
             (np.sqrt(np.sum(np.square((x_t1 - x_t0_mean)), axis=1)) +
              np.sqrt(np.sum(np.square((x_t1 - x_t1_mean)), axis=1)))
    proba0 = np.sqrt(np.sum(np.square((x_t0 - x_t1_mean)), axis=1)) / \
             (np.sqrt(np.sum(np.square((x_t0 - x_t0_mean)), axis=1)) +
              np.sqrt(np.sum(np.square((x_t0 - x_t1_mean)), axis=1)))

    proba1[proba1 == 0] = 1e-8
    proba0[proba0 == 0] = 1e-8

    weight1 = 1. / proba1
    weight0 = 1. / proba0

    weights = np.empty_like(t) * 1.
    weights[t == 1] = weight1
    weights[t == 0] = weight0

    weights = np.reshape(weights, [n, 1])
    return weights


def decorrelation_pre(method, x, t):
    if method == "mlp":
        return decorrelation_mlp(x, t)
    elif method == "ed":
        return decorrelation_ed(x, t)
