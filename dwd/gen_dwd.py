import numpy as np
from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y

from dwd.utils import pm1
from dwd.cv import run_cv
from dwd.linear_model import LinearClassifierMixin


class GenDWD(BaseEstimator, LinearClassifierMixin):
    """
    Generalized Distance Weighted Discrimination

    Solves the gDWD problem using the MM algorithm derived in Wang and Zou, 2017.

    Primary reference: Another look at distance-weighted discrimination by Boxiang Wang and Hui Zou, 2017

    Note the tuning parameter lambd is on a different scale the parameter C which is used in the SOCP formulation.

    Parameters
    ----------
    lambd: float
        Tuning parameter for DWD.

    q: float
        Tuning parameter for generalized DWD (the exponent on the margin terms). When q = 1, gDWD is equivalent to DWD.

    implicit_P: bool
        Whether to use the implicit P^{-1} gamma formulation (in the publication) or the explicit computation (in the arxiv version).

    """
    def __init__(self, lambd=1.0, q=1, implicit_P=True):  # TODO: solver args
        self.lambd = lambd
        self.q = q
        self.implicit_P = implicit_P

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

        # formatting
        # TODO: figure out what we should actually check
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype='numeric')

        self.classes_ = np.unique(y)

        P0_eig = self._get_P0_eig()

        # fit DWD
        self.coef_, self.intercept_, self.obj_vals_, self.C_ = \
            solve_gen_dwd(X=X, y=y,
                          lambd=self.lambd, q=self.q,
                          sample_weight=sample_weight,
                          P0_eig=P0_eig,
                          beta_init=None, offset_init=None,
                          obj_tol=1e-5, max_iter=100)

        self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = self.intercept_.reshape(-1)

        return self

    def cv_init(self, X):
        """
        Initializes the object before computing a cross-valiation.
        """
        self._set_P0_eig(X)
        return self

    def _set_P0_eig(self, X):
        """
        Precomputes eigen decomposition of P0 matrix which makes
        cross-validation much faster.
        """
        self._P0_eig = get_P0_eig(X)

    def _get_P0_eig(self):
        if hasattr(self, '_P0_eig'):
            return self._P0_eig
        else:
            return None


class GenDWDCV(BaseEstimator, LinearClassifierMixin):
    """
    Fits Genralized DWD with cross-validation. gDWD cross-validation
    can be significnatly faster if certain quantities are precomputed.

    Parameters
    ----------
    lambd_vals: list of floats
        The lambda values to cross-validate over.

    q_vals: list of floats
        The q-values to cross validate over.

    cv:
        How to perform cross-valdiation. See documetnation in sklearn.model_selection.GridSearchCV.

    scoring:
        What metric to use to score cross-validation. See documetnation in sklearn.model_selection.GridSearchCV.

    """
    def __init__(self,
                 lambd_vals=np.logspace(-2, 2, 10),
                 q_vals=np.logspace(-2, 2, 5),
                 cv=5, scoring='accuracy'):
        self.lambd_vals = lambd_vals
        self.q_vals = q_vals

        self.cv = cv
        self.scoring = scoring

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

        # formatting
        # TODO: figure out what we should actually check
        X, y = check_X_y(X, y, accept_sparse='csr',
                         dtype='numeric')

        self.classes_ = np.unique(y)

        # run cross validation
        params = {'q': self.q_vals, 'lambd': self.lambd_vals}
        best_params, best_score, best_clf, agg_results, all_cv_results = \
            run_cv(clf=GenDWD(),
                   X=X, y=y,
                   params=params,
                   scoring=self.scoring,
                   cv=self.cv,
                   refit_best=True)

        self.best_estimator_ = best_clf

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.agg_cv_results_ = agg_results
        self.all_cv_results_ = all_cv_results

        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_

        return self


def solve_gen_dwd(X, y, lambd, q=1,
                  sample_weight=None,
                  beta_init=None, offset_init=None,
                  implicit_P=True,
                  obj_tol=1e-5, max_iter=100,
                  P0_eig=None):

    """
    Solves the kernel gDWD problem using the MM algorithm derived in Wang and Zou, 2017.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        Input X data.

    y: array-like, (n_samples, )
        The vector of binary class labels.

    lambd: float
        Tuning parameter for DWD.

    q: float
        Tuning parameter for generalized DWD (the exponent on the margin terms). When q = 1, gDWD is equivalent to DWD.

    beta_init, offset_init:
        Initial values to start the optimization algorithm from.

    sample_weight: None, array-like (n_samples,)
        Optional weight for samples.

    implicit_P: bool
        Whether to implicitly calculate P^{-1} gamma. If P0_eig is precomputed
        this is much faster.

    obj_tol: float
        Stopping condition for difference between successive objective
        functions.

    max_iter: int
        Maximum number of iterations to perform.

    P0_eig: None, tuple of (U, D)
        Precomputed eigenvectors and eigenvalues of P0. Optional.
    """

    # argument checking and formatting
    if lambd < 0:
        raise ValueError("Penalty term must be positive; got (lambd=%r)"
                         % lambd)

    if q < 0:
        raise ValueError("Weight term must be positive; got (q=%r)" % q)

    # TODO: add sample weights
    if sample_weight is not None:
        raise NotImplementedError

    X, y = check_X_y(X, y,
                     accept_sparse='csr',
                     dtype='numeric')

    # convert y to +/- 1
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

    # precompute data
    if implicit_P:
        # data needed to implicitly calculate P_inv @ gamma
        if P0_eig is not None:
            U, D = P0_eig
            D = D.ravel()
            assert U.shape == (n_features + 1, n_features + 1)
            assert len(D) == n_features + 1
        else:
            U, D = get_P0_eig(X)

        pi = D + 2 * n_samples * lambd / M

        u1 = U[0, :]
        v = np.multiply(U, 1.0 / pi).dot(u1)
        g = 2 * n_samples * q * lambd / \
            ((q + 1) ** 2 + 2 * n_samples * lambd * sum(v))

    else:
        # explicitly create P_inv matrix
        col_sums = X.sum(axis=0).reshape(-1, 1)
        d = X.T.dot(X) + (2 * n_samples * lambd / M) * np.eye(n_features)
        P = np.bmat([[np.array([[n_samples]]), col_sums.T],
                     [col_sums, d]]).A
        P_inv = np.linalg.inv(P)

    # store objective values
    obj_vals = []
    prev_obj = dwd_obj(X, y, q, lambd, beta, offset)
    obj_vals.append(prev_obj)

    for i in range(max_iter):

        # get step
        if implicit_P:
            beta_step, offset_step = \
                get_step_implicit(X, y, beta, offset, lambd, q,
                                  U, v, g, pi)
        else:
            beta_step, offset_step = \
                get_step_explicit(X, y, beta, offset, lambd, q, P_inv)

        offset = offset - offset_step  # step[0]
        beta = beta - beta_step  # step[1:]

        # stopping condition
        current_obj = dwd_obj(X, y, q, lambd, beta, offset)
        obj_vals.append(current_obj)

        if np.abs(current_obj - prev_obj) < obj_tol:
            break
        else:
            prev_obj = deepcopy(current_obj)

    c = c_from_lambd(lambd, q, beta)

    return beta, offset, obj_vals, c


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


def get_step_implicit(X, y, beta, offset, lambd, q, U, v, g, pi):
    """
    Gets the MM step by implicitly calculating P^{-1} gamma
    """
    n_samples = X.shape[0]

    # compute update
    z = y * V_grad(y * (X.dot(beta) + offset), q=q) / n_samples

    gamma = X.T.dot(z) + 2 * lambd * beta
    gamma = np.insert(gamma, 0, z.sum())

    # implictly compute P_inv @ gamma

    P_inv_gamma = np.multiply(U, 1.0 / pi).dot(U.T.dot(gamma)) + \
        g * v * v.T.dot(gamma)

    # compute step
    step = (n_samples * q / (q + 1) ** 2) * P_inv_gamma

    offset_step = step[0]
    beta_step = step[1:]

    return beta_step, offset_step


def get_step_explicit(X, y, beta, offset, lambd, q, P_inv):
    """
    Gets the MM step by explicitly calculating P^{-1} gamma
    """
    n_samples = X.shape[0]

    # compute update
    z = y * V_grad(y * (X.dot(beta) + offset), q=q) / n_samples

    gamma = X.T.dot(z) + 2 * lambd * beta
    gamma = np.insert(gamma, 0, z.sum())

    step = (n_samples * q / (q + 1) ** 2) * P_inv @ gamma

    offset_step = step[0]
    beta_step = step[1:]

    return beta_step, offset_step


def V_(u, q=1):
    """
    DWD loss function
    """
    if u <= q / (q + 1.0):
        return 1 - u
    else:
        return (1.0 / u ** q) * (q ** q / (q + 1) ** (q + 1))


# vectorized DWD loss function
V = np.vectorize(V_, excluded=['q'])


def V_grad_(u, q=1):
    """
    DWD loss function gradient
    """
    # TODO: check
    if u <= q / (q + 1.0):
        return -1
    else:
        return - (q / (u * (q + 1))) ** (q + 1)
        # return - (1.0 / u ** (q + 1)) * (q / (q + 1)) ** (q + 1)


# vectorized DWD loss gradient
V_grad = np.vectorize(V_grad_, excluded=['q'])


def dwd_obj(X, y, q, lambd, beta, offset):
    """
    DWD objective function.
    """
    return np.mean(V(y * (X.dot(beta) + offset), q=q) + lambd * beta.dot(beta))


def c_from_lambd(lambd, q, beta):
    """
    Gets the tuning paramter, C, for the SOCP formulation of DWD
    from lambda.
    """
    return ((q + 1) ** (q + 1) / q ** q) * np.linalg.norm(beta) ** (q + 1)
