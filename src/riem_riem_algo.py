import numpy as np

from src.utils import *

from pyriemann.utils.distance import distance_riemann, pairwise_distance
from pyriemann.utils.base import sqrtm, invsqrtm, logm
from pyriemann.datasets import sample_gaussian_spd

from sklearn.manifold._utils import _binary_search_perplexity

import matplotlib.pyplot as plt
import warnings
from time import time


class Riem_Riem_tSNE:
    """Riemannian version of t-SNE algorithm. Reduces a set of c x c SPD matrices into a set of 2 x 2 SPD matrices.

    Parameters
    ----------
    perplexity : int, default = None
        Perplexity used in the Riemannian t-SNE algorithm. If perplexity = None, it will be set to 0.75*n_matrices
    verbosity : int, default = 1
        Level of information printed by the optimizer while it operates: 0 is silent, 2 is most verbose.
    max_it : int, default = 10000
        Maximum number of iterations used for the Riemannian gradient descent.
    max_time : int, default = 300
        Maximum time on the run time of the Riemannian gradient descent in seconds.
    initial_point : ndarray, shape (n_matrices, 2, 2), default = None
        The initial point of the Riemannian gradient descent. If None, the initial point is randomly sampled.

    Attributes
    ----------
    res_opti : ndarray, shape (n_matrices, 2, 2)
        The result of the optimization process.
    """

    def __init__(
        self,
        perplexity=None,
        verbosity=1,
        max_it=10000,
        max_time=300,
        initial_point=None,
    ):
        self.perplexity = perplexity
        self.verbosity = verbosity
        self.max_it = max_it
        self.max_time = max_time
        self.initial_point = initial_point
        self.loss_evolution = None
        self.res_opti = None

    def compute_similarties(self, X):
        """Computed the high dimensional symmetrized conditional similarities p_{ij} for the t-SNE algorithm.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices to reduce.

        Returns
        ----------
        P : ndarray, shape (n_matrices, n_matrices)
            The matrix of the symmetrized conditional probabilities of X.
        """
        nb_mat = X.shape[0]
        Dsq = pairwise_distance(X, squared=True)
        Dsq = Dsq.astype(np.float32, copy=False)
        # Use _binary_search_perplexity from sklearn to compute conditional probabilities
        # such that they approximately match the desired perplexity
        conditional_P = _binary_search_perplexity(Dsq, self.perplexity, 0)

        # Symmetrize the conditional probabilities
        P = conditional_P + conditional_P.T
        return P / (2 * nb_mat)

    def compute_low_affinities(self, Y):
        """Computed the low dimensional similarities q_{ij} for the t-SNE algorithm.

        Parameters
        ----------
        Y : ndarray, shape (n_matrices, 2, 2)
            Set of SPD matrices.

        Returns
        ----------
        Q : ndarray, shape (n_matrices, n_matrices)
            The matrix of the low dimensional similarities conditional probabilities of Y.
        Dsq : ndarray, shape (n_matrices, n_matrices)
            The array containing the squared Riemannian distances between the points in X.
        """
        nb_mat = Y.shape[0]
        Dsq = pairwise_distance(Y, squared=True)

        denominator = np.sum(
            [np.sum([np.delete(1 / (1 + Dsq[k, :]), k)]) for k in range(nb_mat)]
        )
        Q = 1 / (1 + Dsq) / denominator
        np.fill_diagonal(Q, 0)
        return Q, Dsq

    def cost(self, P, Q):
        """Computed the loss of the t-SNE, that is the Kullback-Leibler divergence between P and Q.

        Parameters
        ----------
        P : ndarray, shape (n_matrices, n_matrices)
            The matrix of the symmetrized conditional probabilities of X.
        Q : ndarray, shape (n_matrices, n_matrices)
            The matrix of the low dimensional similarities conditional probabilities of Y.

        Returns
        ----------
        _ : float
            The cost of the t-SNE.
        """
        return np.sum(P * np.log((P + np.eye(P.shape[0])) / (Q + np.eye(P.shape[0]))))

    def riemannian_gradient(self, Y, P, Q, Dsq):
        """Computed the Riemannian gradient of the loss of the t-SNE.

        Parameters
        ----------
        Y : ndarray, shape (n_matrices, 2, 2)
            Set of SPD matrices.
        P : ndarray, shape (n_matrices, n_matrices)
            The matrix of the symmetrized conditional probabilities of X.
        Q : ndarray, shape (n_matrices, n_matrices)
            The matrix of the low dimensional similarities conditional probabilities of Y.
        Dsq : ndarray, shape (n_matrices, n_matrices)
            The Riemannian distance matrix of Y.

        Returns
        ----------
        grad : ndarray, shape (n_matrices, 2, 2)
            The Riemannian gradient of the cost of the t-SNE.
        """
        nb_mat = P.shape[0]
        grad = np.zeros((nb_mat, 2, 2))
        Y_i_invsqrt = invsqrtm(Y)
        Y_i_sqrt = sqrtm(Y)
        for i in range(nb_mat):
            log_riemann = (
                Y_i_sqrt[i] @ logm(Y_i_invsqrt[i] @ Y @ Y_i_invsqrt[i]) @ Y_i_sqrt[i]
            )
            grad[i] = -4 * np.sum(
                ((P[i] - Q[i]) / (1 + Dsq[i]))[:, np.newaxis, np.newaxis] * log_riemann,
                axis=0,
            )
        return grad

    def run_minimization(self, P):
        """Run the minimization to solve the t-SNE optimization.

        Parameters
        ----------
        P : ndarray, shape (n_matrices, n_matrices)
            The matrix of the symmetrized conditional probabilities of X.

        Returns
        ----------
        current_sol : ndarray, shape (n_matrices, 2, 2)
            The solution of the t-SNE problem.
        """
        tol_step = 1e-6
        current_sol = self.initial_point
        self.loss_evolution = []
        initial_time = time()

        # loop over iterations
        for i in range(self.max_it):
            if self.verbosity >= 2 and i % 100 == 0:
                print("Iteration : ", i)

            # get the current value for the loss function
            Q, Dsq = self.compute_low_affinities(current_sol)
            loss = self.cost(P, Q)
            self.loss_evolution.append(loss)

            # get the direction of steepest descent
            direction = self.riemannian_gradient(current_sol, P, Q, Dsq)
            norm_direction = norm_SPD(current_sol, direction)

            # backtracking line search
            if i == 0:
                alpha = 1.0 / norm_direction
            else:
                # Pick initial step size based on where we were last time and look a bit further
                # See Boumal, 2023, Section 4.3 for more insights.
                alpha = 4 * (self.loss_evolution[-2] - loss) / (norm_direction**2)

            tau = 0.50
            r = 1e-4
            maxiter_linesearch = 25

            retracted = retraction(current_sol, -alpha * direction)
            Q_retracted, Dsq_retracted = self.compute_low_affinities(retracted)
            loss_retracted = self.cost(P, Q_retracted)

            # Backtrack while the Armijo criterion is not satisfied
            for _ in range(maxiter_linesearch):
                if loss - loss_retracted > r * alpha * norm_direction**2:
                    break
                alpha = tau * alpha

                retracted = retraction(current_sol, -alpha * direction)
                Q_retracted, Dsq_retracted = self.compute_low_affinities(retracted)
                loss_retracted = self.cost(P, Q_retracted)
            else:
                print("Maximum iteration in linesearched reached.")

            # update variable for next iteration
            current_sol = retracted

            # test if the step size is small
            crit = norm_SPD(current_sol, -alpha * direction)
            if crit <= tol_step:
                print("Min stepsize reached")
                break

            # test if the maximum time has been reached
            if time() - initial_time >= self.max_time:
                warnings.warn("Time limite reached after " + str(i) + " iterations.")
                break

        else:
            warnings.warn("Maximum iterations reached.")
        print(
            "Optimization done in "
            + str(np.round(time() - initial_time, 2))
            + " seconds."
        )
        return current_sol

    def fit(self, X):
        """Fit X to 2x2 SDP matrices using the Riemannian t-SNE algorithm.

        Parameters
        ----------
        X : array_like of shape (n_matrices, n_channels, n_channels)

        Returns
        ----------
        res_opti :  ndarray, shape (n_matrices, 2, 2)
            The solution of the t-SNE problem.
        """
        (n_matrices, _, _) = X.shape

        if self.perplexity is None:
            self.perplexity = int(0.75 * n_matrices)

        # Compute similarities in the high dimension space
        P = self.compute_similarties(X)

        if self.initial_point is None:
            # Sample initial solution close to the identity
            sigma = 1e-2
            self.initial_point = sample_gaussian_spd(
                n_matrices, mean=np.eye(2), sigma=sigma
            )
        if self.verbosity >= 1:
            print("Optimizing...")
        self.res_opti = self.run_minimization(P)

        return self.res_opti

    def plot_loss(self):
        """Plot the evolution of the loss during the Riemannian gradient descent in log-log scale."""

        if self.loss_evolution is None:
            raise Exception("You need to fit the T-SNE before plotting the loss.")
        plt.figure("Plot of the loss")
        plt.loglog(self.loss_evolution)
        plt.xlabel("Log of the iterations")
        plt.ylabel("log of the loss")
        plt.title("Log-log plot of the evolution of the loss")
        plt.show()


class Riem_Riem_MDS:
    """Riemannian version of MDS algorithm. Reduces a set of c x c SPD matrices into a set of 2 x 2 SPD matrices.

    Parameters
    ----------
    verbosity : int, default = 1
        Level of information printed by the optimizer while it operates: 0 is silent, 2 is most verbose.
    max_it : int, default = 10000
        Maximum number of iterations used for the Riemannian gradient descent.
    max_time : int, default = 300
        Maximum time on the run time of the Riemannian gradient descent in seconds.
    initial_point : ndarray, shape (n_matrices, 2, 2), default = None
        The initial point of the Riemannian gradient descent. If None, the initial point is randomly sampled.

    Attributes
    ----------
    res_opti : ndarray, shape (n_matrices, 2, 2)
        The result of the optimization process.
    """

    def __init__(self, verbosity=1, max_it=10000, max_time=300, initial_point=None):
        self.verbosity = verbosity
        self.max_it = max_it
        self.max_time = max_time
        self.initial_point = initial_point
        self.res_opti = None
        self.loss_evolution = None

    def cost(self, Z, D):
        """Computed the loss of the MDS

        Parameters
        ----------
        Z : ndarray, shape (n_matrices, 2, 2)
            Set of 2 x 2 SPD matrices
        D : ndarray, shape (n_matrices, n_matrices)
            The matrix of the pairwise Riemannian distances for the high dimensional set of SPD matrices.

        Returns
        ----------
        _ : float
            The cost of the MDS.
        """
        N = Z.shape[0]
        D_Z = pairwise_distance(Z)
        return np.sum(((D_Z - D) ** 2)[np.tril_indices(N)])

    def riemannian_gradient(self, Z, D):
        """Computed the Riemannian gradient of the loss of the MDS.

        Parameters
        ----------
        Z : ndarray, shape (n_matrices, 2, 2)
            Set of 2 x 2 SPD matrices
        D : ndarray, shape (n_matrices, n_matrices)
            The matrix of the pairwise Riemannian distances for the high dimensional set of SPD matrices.

        Returns
        ----------
        g : ndarray, shape (n_matrices, 2, 2)
            The Riemannian gradient of the cost of the MDS.
        """

        N = Z.shape[0]
        g = np.zeros((N, 2, 2))
        Z_k_invsqrt = invsqrtm(Z)
        Z_k_sqrt = sqrtm(Z)
        D_Z = pairwise_distance(Z) + np.eye(N)
        for k in range(N):
            log_riemann = (
                Z_k_sqrt[k] @ logm(Z_k_invsqrt[k] @ Z @ Z_k_invsqrt[k]) @ Z_k_sqrt[k]
            )
            coeff = D[k] / D_Z[:, k] - 1
            coeff[k] = 0
            g[k] = 2 * np.sum(
                coeff[:, np.newaxis, np.newaxis] * log_riemann, axis=0
            )  # ,where=np.delete(np.arange(N),k))
        return (g + np.transpose(g, (0, 2, 1))) / 2

    def run_minimization(self, D):
        """Run the minimization to solve the MDS optimization.

        Parameters
        ----------
        D : ndarray, shape (n_matrices, n_matrices)
            The matrix of the pairwise Riemannian distances of X.
        Returns
        ----------
        current_sol : ndarray, shape (n_matrices, 2, 2)
            The solution of the MDS problem.
        """

        tol_step = 1e-6
        current_sol = self.initial_point
        self.loss_evolution = []
        initial_time = time()
        # loop over iterations
        for i in range(self.max_it):
            if self.verbosity >= 2 and i % 100 == 0:
                print("Iteration : ", i)

            # get the current value for the loss function
            loss = self.cost(current_sol, D)
            self.loss_evolution.append(loss)

            # get the direction of steepest descent
            direction = self.riemannian_gradient(current_sol, D)
            norm_direction = norm_SPD(current_sol, direction)

            # backtracking line search
            if i == 0:
                alpha = 1.0 / norm_direction
            else:
                # Pick initial step size based on where we were last time and look a bit further
                # See Boumal, 2023, Section 4.3 for more insights.
                alpha = 4 * (self.loss_evolution[-2] - loss) / (norm_direction**2)

            tau = 0.50
            r = 1e-4
            maxiter_linesearch = 25

            retracted = retraction(current_sol, -alpha * direction)

            loss_retracted = self.cost(retracted, D)
            for _ in range(maxiter_linesearch):
                # Backtrack while the Armijo criterion is not satisfied
                if loss - loss_retracted > r * alpha * norm_direction**2:
                    break
                alpha = tau * alpha
                retracted = retraction(current_sol, -alpha * direction)
                loss_retracted = self.cost(retracted, D)
            else:
                print("Maximum iteration in linesearched reached.")

            # update variable for next iteration
            current_sol = retracted

            # test if the step size is small
            crit = norm_SPD(current_sol, -alpha * direction)
            if crit <= tol_step:
                print("Min stepsize reached")
                break

            # test if the maximum time has been reached
            if time() - initial_time >= self.max_time:
                warnings.warn("Time limite reached after " + str(i) + " iterations.")
                break

        else:
            warnings.warn("Maximum iterations reached.")
        print(
            "Optimization done in "
            + str(np.round(time() - initial_time, 2))
            + " seconds."
        )
        return current_sol

    def fit(self, X):
        """Fit X to 2x2 SDP matrices using the Riemannian MDS algorithm.

        Parameters
        ----------
        X : array_like of shape (n_matrices, n_channels, n_channels)

        Returns
        ----------
        res_opti :  ndarray, shape (n_matrices, 2, 2)
            The solution of the MDS problem.
        """
        (nb_mat, dim, _) = X.shape
        D = pairwise_distance(X)

        if self.initial_point is None:
            # Sample initial solution close to the identity
            sigma = 1e-2
            self.initial_point = sample_gaussian_spd(
                nb_mat, mean=np.eye(2), sigma=sigma
            )

        if self.verbosity >= 1:
            print("Optimizing...")
        self.res_opti = self.run_minimization(D)

        return self.res_opti

    def plot_loss(self):
        """Plot the evolution of the loss during the Riemannian gradient descent in log-log scale."""
        if self.res_opti is None:
            raise Exception("You need to fit the MDS before plotting the loss.")
        plt.figure("Plot of the loss")
        plt.loglog(self.loss_evolution)
        plt.xlabel("Log of the iterations")
        plt.ylabel("log of the loss")
        plt.title("Log-log plot of the evolution of the loss")
        plt.show()
