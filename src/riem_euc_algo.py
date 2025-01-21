import numpy as np

from src.utils import *

from pyriemann.utils.distance import distance_riemann, pairwise_distance
from pyriemann.utils.base import sqrtm, invsqrtm, logm
from pyriemann.datasets import sample_gaussian_spd

import pymanopt
from pymanopt.manifolds import SymmetricPositiveDefinite
from pymanopt.optimizers import SteepestDescent

from sklearn.manifold._utils import _binary_search_perplexity

import matplotlib.pyplot as plt
import warnings
from time import time
from scipy.spatial.distance import cdist
from scipy.stats import norm


class Riem_Euc_tSNE:
    """Euclidean version of t-SNE that reduces the points in R^3 equipped with its classical Euclidean structure
    We compute the initial distances using the Riemannian distance"""

    def __init__(
        self, perplexity, verbosity, max_it=10000, max_time=1000, initial_point=None
    ):
        self.perplexity = perplexity
        self.verbosity = verbosity
        self.max_it = max_it
        self.max_time = max_time
        self.initial_point = initial_point
        self.res_opti = None

    def compute_distances(self, X):
        nb_mat = X.shape[0]
        D = np.zeros((nb_mat, nb_mat))
        for i in range(nb_mat):
            for j in range(i + 1, nb_mat):
                dist = np.linalg.norm(X[i] - X[j])
                D[i, j] = dist
                D[j, i] = dist
        return D**2

    def compute_similarties(self, X, perplexity):
        nb_mat = X.shape[0]
        Dsq = pairwise_distance(X, squared=True)
        Dsq = Dsq.astype(np.float32, copy=False)
        conditional_P = _binary_search_perplexity(Dsq, perplexity, 0)
        P = conditional_P + conditional_P.T
        return P / (2 * nb_mat)

    def compute_low_affinities(self, Y):
        nb_mat = Y.shape[0]
        Y_ = np.reshape(Y, (nb_mat, 3))
        Dsq = cdist(Y_, Y_) ** 2

        denom = np.sum(
            [np.sum([np.delete(1 / (1 + Dsq[k, :]), k)]) for k in range(nb_mat)]
        )
        Q = (1 / (1 + Dsq)) / denom
        np.fill_diagonal(Q, 0)
        return Q, Dsq

    def create_cost_tsne(self, manifold, P, nb_mat):
        nb_mat = P.shape[0]

        @pymanopt.function.numpy(manifold)
        def cost(Y):
            Y_ = np.reshape(Y, (nb_mat, 3))
            Q, D = self.compute_low_affinities(Y_)
            return np.sum(
                P * np.log((P + np.eye(P.shape[0])) / (Q + np.eye(P.shape[0])))
            )

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(Y):
            Y_ = np.reshape(Y, (nb_mat, 3))
            Q, Dsq = self.compute_low_affinities(Y_)
            grad = np.zeros((nb_mat, 3))
            directions = Y_[:, np.newaxis, :] - Y_
            for i in range(nb_mat):
                grad[i] = 4 * np.sum(
                    ((P[i] - Q[i]) / (1 + Dsq[i]))[:, np.newaxis] * directions[i],
                    axis=0,
                )
            return grad.flatten()

        return cost, euclidean_gradient

    def fit(self, X):
        (nb_mat, dim, _) = X.shape
        manifold = pymanopt.manifolds.Euclidean(nb_mat * 3)

        P = self.compute_similarties(X, self.perplexity)

        if self.initial_point is None:
            sigma = 1e-4
            self.initial_point = norm.rvs(scale=sigma, size=nb_mat * 3)

        cost, euclidean_gradient = self.create_cost_tsne(manifold, P, nb_mat)
        problem = pymanopt.Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient
        )
        optimizer = SteepestDescent(
            verbosity=self.verbosity,
            log_verbosity=2,
            max_iterations=self.max_it,
            max_time=self.max_time,
        )
        res_final = optimizer.run(problem, initial_point=self.initial_point)
        self.res_opti = res_final

        return res_final

    def plot_loss(self):
        if self.res_opti is None:
            raise Exception("You need to fit the T-SNE before plotting the loss.")
        plt.figure("Plot of the loss")
        plt.loglog(self.res_opti.log["iterations"]["cost"])
        plt.xlabel("Log of the iterations")
        plt.ylabel("log of the loss")
        plt.title("Log-log plot of the evolution of the loss")
        plt.show()


class Riem_Euc_MDS:
    """Riemannian - Euclidean version of MDS that reduces the points in R^3 equipped with its classical Euclidean structure
    by computing the initial distances using the Riemannian metric.
    """

    def __init__(self, verbosity, max_it=10000, max_time=1000, initial_point=None):
        self.verbosity = verbosity
        self.max_it = max_it
        self.max_time = max_time
        self.initial_point = initial_point
        self.res_opti = None

    def create_cost_mds(self, manifold_opti, D):
        N = D.shape[0]

        @pymanopt.function.numpy(manifold_opti)
        def cost(Z):
            Z_ = np.reshape(Z, (N, 3))
            D_Z = cdist(Z_, Z_)
            return np.sum(((D_Z - D) ** 2)[np.tril_indices(N)])

        @pymanopt.function.numpy(manifold_opti)
        def euclidean_gradient(Z):
            Z_ = np.reshape(Z, (N, 3))
            D_Z = cdist(Z_, Z_) + np.eye(N)
            g = np.zeros((N, 3))
            directions = Z_[:, np.newaxis, :] - Z_
            for k in range(N):
                coeff = 1 - D[k] / D_Z[:, k]
                g[k] = 2 * np.sum(coeff[:, np.newaxis] * directions[k], axis=0)
            return g.flatten()

        return cost, euclidean_gradient

    def fit(self, X):
        (nb_mat, dim, _) = X.shape
        D = pairwise_distance(X)
        manifold = pymanopt.manifolds.Euclidean(3 * nb_mat)
        cost, euclidean_gradient = self.create_cost_mds(manifold, D)

        problem = pymanopt.Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient
        )
        optimizer = SteepestDescent(
            verbosity=self.verbosity,
            log_verbosity=2,
            max_iterations=self.max_it,
            max_time=self.max_time,
        )
        res_final = optimizer.run(problem, initial_point=self.initial_point)

        self.res_opti = res_final
        return res_final

    def plot_loss(self):
        if self.res_opti is None:
            raise Exception("You need to fit the MDS before plotting the loss.")
        plt.figure("Plot of the loss")
        plt.loglog(self.res_opti.log["iterations"]["cost"])
        plt.xlabel("Log of the iterations")
        plt.ylabel("log of the loss")
        plt.title("Log-log plot of the evolution of the loss")
        plt.show()
