# all the imports needed for this blog
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib
from sklearn.datasets import make_blobs
import torch
matplotlib.use("WebAgg")

x, target = make_blobs(200)
plt.scatter(x[:, 0], x[:, 1], c=target)
#plt.show()


def similarity(xi, xj):
    return -((xi - xj) ** 2).sum()


def create_matrices():
    S = torch.zeros((x.shape[0], x.shape[0]))
    R = S.clone().detach()
    A = S.clone().detach()

    # compute similarity for every data point.
    for i in range(x.shape[0]):
        for k in range(x.shape[0]):
            S[i, k] = similarity(x[i], x[k])

    return A, R, S


def update_r(damping=0.9):
    global R
    # For every column k, except for the column with the maximum value the max is the same.
    # So we can subtract the maximum for every row,
    # and only need to do something different for k == argmax

    v = S + A
    rows = np.arange(x.shape[0])
    # We only compare the current point to all other points,
    # so the diagonal can be filled with -infinity
    #np.fill_diagonal(v, -np.inf)
    v.fill_diagonal_(-np.inf)

    # max values
    idx_max = np.argmax(v, axis=1)
    first_max = v[rows, idx_max]

    # Second max values. For every column where k is the max value.
    v[rows, idx_max] = -np.inf
    second_max = v[rows, np.argmax(v, axis=1)]

    # Broadcast the maximum value per row over all the columns per row.
    max_matrix = torch.zeros_like(R) + first_max[:, None]
    max_matrix[rows, idx_max] = second_max

    new_val = S - max_matrix

    R = R * damping + (1 - damping) * new_val


def update_a(damping=0.9):
    global A

    k_k_idx = np.arange(x.shape[0])
    # set a(i, k)
    a = R.clone().detach()
    a[a < 0] = 0
    #np.fill_diagonal(a, 0)
    a.fill_diagonal_(0)
    a = a.sum(axis=0) # columnwise sum
    a = a + R[k_k_idx, k_k_idx]

    # broadcasting of columns 'r(k, k) + sum(max(0, r(i', k))) to rows.
    a = torch.ones(A.shape) * a

    # For every column k, subtract the positive value of k.
    # This value is included in the sum and shouldn't be
    a -= np.clip(R, 0, np.inf)
    a[a > 0] = 0

    # set(a(k, k))
    w = R.clone().detach()
    #np.fill_diagonal(w, 0)
    w.fill_diagonal_(0)

    w[w < 0] = 0

    a[k_k_idx, k_k_idx] = w.sum(axis=0) # column wise sum
    A = A * damping + (1 - damping) * a



def plot_iteration(A, R):
    fig = plt.figure(figsize=(12, 6))
    sol = A + R
    # every data point i chooses the maximum index k
    labels = np.argmax(sol, axis=1)
    exemplars = np.unique(labels)
    colors = dict(zip(exemplars, cycle('bgrcmyk')))

    for i in range(len(labels)):
        X = x[i][0]
        Y = x[i][1]

        if i in exemplars:
            exemplar = i
            edge = 'k'
            ms = 10
        else:
            exemplar = labels[i].item()
            ms = 3
            edge = None
            plt.plot([X, x[exemplar][0]], [Y, x[exemplar][1]], c=colors[exemplar])
        plt.plot(X, Y, 'o', markersize=ms, markeredgecolor=edge, c=colors[exemplar])

    plt.title('Number of exemplars: %s' % len(exemplars))
    return fig, labels, exemplars


A, R, S = create_matrices()
preference = np.median(S)
#np.fill_diagonal(S, preference)
S.fill_diagonal_(preference)
damping = 0.5

figures = []
last_sol = np.ones(A.shape)
last_exemplars = np.array([])

c = 0
for i in range(200):
    update_r(damping)
    update_a(damping)

    sol = A + R
    exemplars = np.unique(np.argmax(sol, axis=1))

    if last_exemplars.size != exemplars.size or np.all(last_exemplars != exemplars):
        fig, labels, exemplars = plot_iteration(A, R)
        figures.append(fig)

    if np.allclose(last_sol, sol):
        print(exemplars, i)
        break

    last_sol = sol
    last_exemplars = exemplars
plt.show()