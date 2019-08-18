import numpy as np
from copy import deepcopy

from sklearn.base import BaseEstimator  # , TransformerMixin, ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets

from dwd.utils import pm1


class GenDWD(BaseEstimator, LinearClassifierMixin):

    def __init__(self, lambd=1.0, q=1, algo='implicit_P'):  # TODO: solver args
        self.lambd = lambd
        self.q = q
        self.algo = algo

    def fit(self, X, y, sample_weight=None, P0_eig=None):
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

        P0: None, (U, D)
            Precomputed P0 matrix (optional)

        Returns
        -------
        self : object
        """
        # TODO: what to do about multi-class

        if self.lambd < 0:
            raise ValueError("Penalty term must be positive; got (lambd=%r)"
                             % self.lambd)

        if self.q < 0:
            raise ValueError("Weight term must be positive; got (q=%r)"
                             % self.q)

        # formatting
        # TODO: figure out what we should actually check
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype=np.float64, order="C",
                         accept_large_sparse=False)
        check_classification_targets(y)

        self.classes_ = np.unique(y)

        # fit DWD
        if self.algo == 'implicit_P':
            coef, intercept, obj_vals, C = \
                solve_gen_dwd_implicit_P(X=X, y=y,
                                         lambd=self.labmd, q=self.q,
                                         sample_weight=sample_weight,
                                         P0_eig=P0_eig,
                                         beta_init=None, offset_init=None,
                                         obj_tol=1e-5, max_iter=100)

        elif self.algo == 'explicit_P':
            coef, intercept, obj_vals, C = \
                solve_gen_dwd_explicit_P(X=X, y=y,
                                         lambd=self.labmd, q=self.q,
                                         sample_weight=sample_weight,
                                         beta_init=None, offset_init=None,
                                         obj_tol=1e-5, max_iter=100)

        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = intercept.reshape(-1)
        self.obj_vals_ = obj_vals
        self.C_ = C

        return self


def get_P0_eig(X):
    """
    Equation (3.6)

    Parameters
    ----------
    X: (n_samples, n_features)

    Output
    ------
    U, D

    U: (n_features, n_features)
        Eigenvectors of P0.

    D: (n_features, )
        Eigenvalues of P0 in decending order.
    """

    n = np.array([[X.shape[0]]])
    colsum = X.sum(axis=0).reshape(-1, 1)

    # create P0
    P0 = [[n, colsum.T],
          [colsum, X.T.dot(X)]]  # (d+1 x d+1)
    P0 = np.bmat(P0).A

    # compute eigen decomp of P0
    D, U = np.linalg.eigh(P0)
    D = D[::-1]  # sort evals in decending order
    U = U[:, ::-1]

    return U, D


def solve_gen_dwd_implicit_P(X, y, lambd, q=1,
                             sample_weight=None,
                             beta_init=None, offset_init=None,
                             obj_tol=1e-5, max_iter=100,
                             P0_eig=None):
    """
    Algorithm 1 from https://arxiv.org/pdf/1508.05913.pdf
    """
    # TODO: add sample weights
    if sample_weight is not None:
        raise NotImplementedError

    X = np.array(X)
    y = pm1(y)

    n_samples, n_features = X.shape
    M = (q + 1) ** 2 / q

    # initialize variables
    if beta_init is None:
        beta = np.random.normal(size=n_features)
    else:
        beta = beta_init

    if offset_init is None:
        offset = 0.0
    else:
        offset = offset_init

    # get eigendecomp of P0
    if P0_eig is not None:
        U, D = P0_eig
        D = D.ravel()
        assert U.shape == (n_features, n_features)
        assert len(D) == n_features
    else:
        U, D = get_P0_eig(X)

    pi = D + 2 * n_samples * lambd / M

    # store objective values
    obj_vals = []
    prev_obj = dwd_obj(X, y, q, lambd, beta, offset)
    obj_vals.append(prev_obj)

    for i in range(max_iter):
        # compute update
        z = y * V_grad(y * (X.dot(beta) + offset), q=q) / n_samples

        gamma = X.T.dot(z) + 2 * lambd * beta
        gamma = np.insert(gamma, 0, z.sum())

        # implictly compute P_inv @ gamma
        # TODO: get rid of explicit diagnaol matrices
        u1 = U[0, :]
        v = U.dot(np.diag(1.0 / pi)).dot(u1)
        g = 2 * n_samples * q * lambd / \
            ((q + 1) ** 2 + 2 * n_samples * lambd * sum(v))

        P_inv_gamma = U.dot(np.diag(1.0 / pi)).dot(U.T.dot(gamma)) + \
            g * v * v.T.dot(gamma)

        # compute step
        step = (n_samples * q / (q + 1) ** 2) * P_inv_gamma

        # take step
        offset = offset - step[0]
        beta = beta - step[1:]

        # stopping condition
        current_obj = dwd_obj(X, y, q, lambd, beta, offset)
        obj_vals.append(current_obj)

        if np.abs(current_obj - prev_obj) < obj_tol:
            break
        else:
            prev_obj = deepcopy(current_obj)

    c = c_from_lambd(lambd, q, beta)

    return beta, offset, obj_vals, c


def get_step_implicit(X, y, beta, offset, lambd, q, U, pi):
    n_samples = X.shape[0]

    # compute update
    z = y * V_grad(y * (X.dot(beta) + offset), q=q) / n_samples

    gamma = X.T.dot(z) + 2 * lambd * beta
    gamma = np.insert(gamma, 0, z.sum())

    # implictly compute P_inv @ gamma
    # TODO: get rid of explicit diagnaol matrices
    u1 = U[0, :]
    v = U.dot(np.diag(1.0 / pi)).dot(u1)
    g = 2 * n_samples * q * lambd / \
        ((q + 1) ** 2 + 2 * n_samples * lambd * sum(v))

    P_inv_gamma = U.dot(np.diag(1.0 / pi)).dot(U.T.dot(gamma)) + \
        g * v * v.T.dot(gamma)

    # compute step
    step = (n_samples * q / (q + 1) ** 2) * P_inv_gamma

    return step


def get_step_explicit(X, y, beta, offset, lambd, q, P_inv):
    n_samples = X.shape[0]

    # compute update
    z = y * V_grad(y * (X.dot(beta) + offset), q=q) / n_samples

    gamma = X.T.dot(z) + 2 * lambd * beta
    gamma = np.insert(gamma, 0, z.sum())

    step = (n_samples * q / (q + 1) ** 2) * P_inv @ gamma

    return step


def solve_gen_dwd_explicit_P(X, y, lambd, q=1,
                             sample_weight=None,
                             beta_init=None, offset_init=None,
                             obj_tol=1e-5, max_iter=100):
    """
    Algorithm 1 from https://arxiv.org/pdf/1508.05913.pdf
    """
    # TODO: add sample weights
    if sample_weight is not None:
        raise NotImplementedError

    X = np.array(X)
    y = pm1(y)

    n_samples, n_features = X.shape
    M = (q + 1) ** 2 / q

    # create P_inv matrix
    col_sums = X.sum(axis=0).reshape(-1, 1)
    d = X.T.dot(X) + (2 * n_samples * lambd / M) * np.eye(n_features)
    P = np.bmat([[np.array([[n_samples]]), col_sums.T],
                 [col_sums, d]]).A
    P_inv = np.linalg.inv(P)

    # initialize variables
    if beta_init is None:
        beta = np.random.normal(size=n_features)
    else:
        beta = beta_init

    if offset_init is None:
        offset = 0.0
    else:
        offset = offset_init

    # store objective values
    obj_vals = []
    prev_obj = dwd_obj(X, y, q, lambd, beta, offset)
    obj_vals.append(prev_obj)

    for i in range(max_iter):
        # compute update
        z = y * V_grad(y * (X.dot(beta) + offset), q=q) / n_samples

        gamma = X.T.dot(z) + 2 * lambd * beta
        gamma = np.insert(gamma, 0, z.sum())

        step = (n_samples * q / (q + 1) ** 2) * P_inv @ gamma

        # take step
        offset = offset - step[0]
        beta = beta - step[1:]

        # stopping condition
        current_obj = dwd_obj(X, y, q, lambd, beta, offset)
        obj_vals.append(current_obj)

        if np.abs(current_obj - prev_obj) < obj_tol:
            break
        else:
            prev_obj = deepcopy(current_obj)

    c = c_from_lambd(lambd, q, beta)

    return beta, offset, obj_vals, c


def V_(u, q=1):
    if u <= q / (q + 1.0):
        return 1 - u
    else:
        return (1.0 / u ** q) * (q ** q / (q + 1) ** (q + 1))


V = np.vectorize(V_, excluded=['q'])


def V_grad_(u, q=1):
    # TODO: check
    if u <= q / (q + 1.0):
        return -1
    else:
        return - (1.0 / u ** (q + 1)) * (q / (q + 1)) ** (q + 1)


V_grad = np.vectorize(V_grad_, excluded=['q'])


def dwd_obj(X, y, q, lambd, beta, offset):
    return np.mean(V(y * (X.dot(beta) + offset), q=q) + lambd * beta.dot(beta))


def c_from_lambd(lambd, q, beta):
    return ((q + 1) ** (q + 1) / q ** q) * np.linalg.norm(beta) ** (q + 1)
