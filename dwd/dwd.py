import cvxpy as cp
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.metrics.pairwise import euclidean_distances

from dwd.utils import pm1
from dwd.linear_model import LinearClassifierMixin


class DWD(BaseEstimator, LinearClassifierMixin):

    def __init__(self, C=1.0, solver_kws={}):
        self.C = C
        self.solver_kws = solver_kws

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X

        sample_weight : array-like, shape = [n_samples], optional
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

        Returns
        -------
        self : object
        """
        # TODO: what to do about multi-class

        self.classes_ = np.unique(y)

        if self.C == 'auto':
            self.C = auto_dwd_C(X, y)

        # fit DWD
        self.coef_, self.intercept_, self.eta_, self.d_, self.problem_ = \
            solve_dwd_socp(X, y, C=self.C,
                           sample_weight=sample_weight,
                           solver_kws=self.solver_kws)

        self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = self.intercept_.reshape(-1)

        return self


def solve_dwd_socp(X, y, C=1.0, sample_weight=None, solver_kws={}):
    """
    Solves distance weighted discrimination optimization problem.

    Solves problem (2.7) from https://arxiv.org/pdf/1508.05913.pdf

    Parameters
    ----------
    X: (n_samples, n_features)

    y: (n_samples, )

    C: float
        Strictly positive tuning parameter.

    sample_weight: None, (n_samples, )
        Weights for samples.

    solver_kws: dict
        Keyword arguments to cp.solve

    Output
    ------
    beta, intercept, eta, d, problem

    beta: (n_features, )
        DWD normal vector.

    intercept: float
        DWD intercept.

    eta, d:
        optimization variables.

    problem: cp.Problem

    """

    if C < 0:
        raise ValueError("Penalty term must be positive; got (C={})".format(C))

    # TODO: add sample weights
    if sample_weight is not None:
        raise NotImplementedError

    X, y = check_X_y(X, y,
                     accept_sparse='csr',
                     dtype='numeric')

    # convert y to +/- 1
    y = pm1(y)

    n_samples, n_features = X.shape

    # problem data
    X = cp.Parameter(shape=X.shape, value=X)
    y = cp.Parameter(shape=y.shape, value=y)
    C = cp.Parameter(value=C, nonneg=True)

    # optimization variables
    beta = cp.Variable(shape=n_features)
    X_beta = cp.Variable(shape=n_samples)  # to prevent `Y_tilde @ X @ beta` breaking DPP
    intercept = cp.Variable()
    eta = cp.Variable(shape=n_samples, nonneg=True)

    rho = cp.Variable(shape=n_samples)
    sigma = cp.Variable(shape=n_samples)

    # objective funtion
    # TODO: check this is correct way to do sample weighting
    if sample_weight is None:
        v = np.ones(n_samples)
    else:
        v = np.array(sample_weight).reshape(-1)
        assert len(v) == n_samples
    objective = v.T @ (rho + sigma + C * eta)

    # setup constraints
    # TODO: do we need explicit SOCP constraints?
    Y_tilde = cp.diag(y)  # TODO: make sparse
    constraints = [rho - sigma == Y_tilde @ X_beta + intercept * y + eta,
                   cp.SOC(cp.Parameter(value=1), beta),
                   X_beta == X @ beta]  # ||beta||_2^2 <= 1

    # rho^2 - sigma^2 >= 1
    constraints.extend([cp.SOC(rho[i], cp.vstack([sigma[i], 1]))
                        for i in range(n_samples)])

    # solve problem
    problem = cp.Problem(cp.Minimize(objective),
                         constraints=constraints)

    problem.solve(**solver_kws)

    # d = rho - sigma
    # rho = (1/d + d), sigma = (1/d - d)/2
    d = rho.value - sigma.value

    return beta.value, intercept.value, eta.value, d, problem


def auto_dwd_C(X, y, const=100):
    """
    Automatic choice of C from Distance-Weighted Discrimination by Marron et al, 2007. Note this only is for the SOCP formulation of DWD.

    C = 100 / d ** 2

    Where d is the median distance between points in either class.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The input data.

    y: array-like, (n_samples, )
        The vector of binary class labels.

    const: float
        The constanted used to determine C. Originally suggested to be 100.

    """
    labels = np.unique(y)
    assert len(labels) == 2

    # pariwise distances between points in each class
    D = euclidean_distances(X[y == labels[0], :],
                            X[y == labels[1], :])

    d = np.median(D.ravel())

    return const / d ** 2


def dwd_obj(X, y, C, beta, offset, eta):
    """
    Objective function for DWD.
    """
    d = y * (X.dot(beta) + offset) + eta

    return sum(1.0 / d) + C * sum(eta)
