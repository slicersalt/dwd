import numpy as np
from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y

from dwd.utils import pm1
from dwd.gen_dwd import V, V_grad
from dwd.kernel_utils import KernelClfMixin
from dwd.cv import run_cv


class KernGDWD(BaseEstimator, KernelClfMixin):
    """
    Kernel Generalized Distance Weighted Discrimination

    Solves the kernel gDWD problem using the MM algorithm derived in Wang and Zou, 2017.

    Primary reference: Another look at distance-weighted discrimination by Boxiang Wang and Hui Zou, 2017

    Note the tuning parameter lambd is on a different scale the parameter C which is used in the SOCP formulation.

    Parameters
    ----------
    lambd: float
        Tuning parameter for DWD.

    q: float
        Tuning parameter for generalized DWD (the exponent on the margin terms). When q = 1, gDWD is equivalent to DWD.

    kernel: str, callable(X, Y, **kwargs)
        The kernel to use.

    kernel_kws: dict
        Any key word arguments for the kernel.

    implicit_P: bool
        Whether to use the implicit P^{-1} gamma formulation (in the publication) or the explicit computation (in the arxiv version).
    """

    def __init__(self, lambd=1.0, q=1.0, kernel='linear',
                 kernel_kws={}, implicit_P=True):
        self.lambd = lambd
        self.q = q

        self.kernel = kernel
        self.kernel_kws = kernel_kws

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

        self.classes_ = np.unique(y)
        self._Xfit = X  # Store K so we can compute predictions

        K = self._compute_kernel(X)

        K_eig = self._get_K_eig()

        # fit DWD
        alpha, offset, obj_vals, c = \
            solve_gen_kern_dwd(K=K,
                               y=y,
                               lambd=self.lambd,
                               q=self.q,
                               alpha_init=None,
                               offset_init=None,
                               sample_weight=None,
                               implicit_P=self.implicit_P,
                               obj_tol=1e-5, max_iter=100,
                               K_eig=K_eig)

        self.intercept_ = offset.reshape(-1)
        self.dual_coef_ = alpha.reshape(1, -1)

        return self

    def cv_init(self, X):
        """
        Initializes the object before computing a cross-valiation.
        """
        # I don't love that we have to set this here
        self._Xfit = X

        # Warning: we compute the kernel twice -- any way around this
        # without messing up the SKlearn API too badly?
        K = self._compute_kernel(X)
        self._set_K_eig(K)
        return self

    def _set_K_eig(self, X):
        """
        Precomputes eigen decomposition of K matrix which makes
        cross-validation much faster.
        """
        self._K_eig = get_K_eig(X)

    def _get_K_eig(self):
        if hasattr(self, '_K_eig'):
            return self._K_eig
        else:
            return None


class KernGDWDCV(BaseEstimator, KernelClfMixin):
    """
    Fits kernel gDWD with cross-validation. gDWD cross-validation
    can be significnatly faster if certain quantities are precomputed.

    Parameters
    ----------
    lambd_vals: list of floats
        The lambda values to cross-validate over.

    q_vals: list of floats
        The q-values to cross validate over.

    kernel: str, callable
        The kernel to use.

    kern_kws_vals: list of dicts
        The kernel parameters to validate over.

    cv:
        How to perform cross-valdiation. See documetnation in sklearn.model_selection.GridSearchCV.

    scoring:
        What metric to use to score cross-validation. See documetnation in sklearn.model_selection.GridSearchCV.

    """
    def __init__(self,
                 lambd_vals=np.logspace(-2, 2, 10),
                 q_vals=np.logspace(-2, 2, 5),
                 kernel='linear',
                 kernel_kws_vals={},
                 cv=5, scoring='accuracy'):

        self.lambd_vals = lambd_vals
        self.q_vals = q_vals
        self.kernel = kernel
        self.kernel_kws_vals = kernel_kws_vals

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
        params = {'q': self.q_vals, 'lambd': self.lambd_vals,
                  'kernel_kws': self.kernel_kws_vals}

        best_params, best_score, best_clf, agg_results, all_cv_results = \
            run_cv(clf=KernGDWD(kernel=self.kernel),
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

        self._Xfit = self.best_estimator_._Xfit
        self.intercept_ = self.best_estimator_.intercept_
        self.dual_coef_ = self.best_estimator_.dual_coef_

        self.decision_function = self.best_estimator_.decision_function

        return self


def solve_gen_kern_dwd(K, y, lambd, q=1,
                       alpha_init=None, offset_init=None,
                       sample_weight=None,
                       implicit_P=True,
                       obj_tol=1e-5, max_iter=100,
                       K_eig=None):

    """
    Solves the kernel gDWD problem using the MM algorithm derived in Wang and Zou, 2017.

    Parameters
    ----------
    K: array-like, (n_samples, n_samples)
        The kernel.

    y: array-like, (n_samples, )
        The vector of binary class labels.

    lambd: float
        Tuning parameter for DWD.

    q: float
        Tuning parameter for generalized DWD (the exponent on the margin terms). When q = 1, gDWD is equivalent to DWD.

    alpha_init, offset_init:
        Initial values to start the optimization algorithm from.

    sample_weight: None, array-like (n_samples,)
        Optional weight for samples.

    implicit_P: bool
        Whether to use the implicit P^{-1} gamma formulation (in the publication) or the explicit computation (in the arxiv version).

    obj_tol: float
        Stopping condition for difference between successive objective
        functions.

    max_iter: int
        Maximum number of iterations to perform.

    K_eig: None or (U, D)
        Optional. The precomputed eigendecomposition of K.
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

    K, y = check_X_y(K, y,
                     accept_sparse='csr',
                     dtype='numeric')

    assert K.shape[0] == K.shape[1]  # K must be Kernel matrix.

    # convert y to +/- 1
    y = pm1(y)  # convert y to  y +/- 1

    n_samples = K.shape[0]
    M = (q + 1) ** 2 / q

    # precompute data
    if implicit_P:

        # precompute data needed to do implicit P^{-1} gamma
        if K_eig is not None:
            U, Lam = K_eig
            Lam = Lam.ravel()
            assert U.shape == (n_samples, n_samples)
            assert len(Lam) == n_samples
        else:
            U, Lam = get_K_eig(K)

        # see section 4.1 of Wang and Zou, 2017 for details
        pi = Lam ** 2 + (2 * n_samples * lambd / M) * Lam

        ULP = np.multiply(U, Lam * (1.0 / pi))
        v = ULP.dot(U.T.dot(np.ones(n_samples)))

        Ucs = U.sum(axis=0)
        carson = Ucs.dot(np.multiply(ULP, Lam).dot(Ucs))
        g = 1.0 / (n_samples - carson)

    else:
        raise NotImplementedError('explicit_P has a bug ')
        # the objective function explodes
        # TODO: :figure out what is going wrong

        # explicltly calculate P^{-1}
        colsum = K.sum(axis=0).reshape(-1, 1)
        frank = K.dot(K) + (2 * n_samples * lambd / M) * K
        P = np.bmat([[np.array([[n_samples]]), colsum.T],
                     [colsum, frank]]).A
        P_inv = np.linalg.inv(P)

    # initialize variables
    if alpha_init is None:
        alpha = np.random.normal(size=n_samples)
        alpha /= np.linalg.norm(alpha)
    else:
        alpha = alpha_init

    if offset_init is None:
        offset = 0.0
    else:
        offset = offset_init

    # store objective values
    obj_vals = []
    prev_obj = kern_dwd_obj(K, y, q, lambd, alpha, offset)
    obj_vals.append(prev_obj)

    for i in range(max_iter):

        # get step
        if implicit_P:
            alpha_step, offset_step = \
                get_step_implicit_P(K=K, y=y, q=q, lambd=lambd,
                                    alpha=alpha, offset=offset,
                                    U=U, Lam=Lam, pi=pi, v=v, g=g)
        else:
            alpha_step, offset_step = \
                get_step_explicit_P(K=K, y=y, q=q, lambd=lambd,
                                    alpha=alpha, offset=offset,
                                    P_inv=P_inv)

        # update parameters
        alpha = alpha - alpha_step
        offset = offset - offset_step

        # stopping condition
        current_obj = kern_dwd_obj(K, y, q, lambd, alpha, offset)
        obj_vals.append(current_obj)

        if np.abs(current_obj - prev_obj) < obj_tol:
            break
        else:
            prev_obj = deepcopy(current_obj)

    # tuning paramter for SOCP formulation
    c = c_from_lambd(K, lambd, q, alpha)

    return alpha, offset, obj_vals, c


def get_K_eig(K):
    """
    Computes the eigendecomposition of K.

    Parameters
    ----------
    K: (n_samples, n_samples)
        Kernel matrix.

    Output
    ------
    U, D

    U: (n_features, n_features)
        Eigenvectors of K.

    D: (n_features, )
        Eigenvalues of K in decending order.
    """

    Lam, U = np.linalg.eigh(K)
    Lam = Lam[::-1]  # sort evals in decending order
    U = U[:, ::-1]

    return U, Lam


def get_step_implicit_P(K, y, q, lambd, alpha, offset, U, Lam, pi, v, g):
    """
    Computes the step size for one step of the MM algorithm.
    See Wang and Zou, 2017 for details.
    """
    n_samples = K.shape[0]

    z = y * V_grad(y * (K.dot(alpha) + offset), q=q) / n_samples

    alice = z + 2 * lambd * alpha

    bob = np.insert(-v, 0, values=1)
    bob = (sum(z) - z.T.dot(K.dot(alice))) * bob

    cathy = np.multiply(U, Lam * (1.0 / pi)).dot(U.T.dot(alice))
    cathy = np.insert(cathy, 0, 0)

    P_inv_gamma = g * bob + cathy

    step = (n_samples * q / ((q + 1) ** 2)) * P_inv_gamma

    offset_step = step[0]
    alpha_step = step[1:]

    return alpha_step, offset_step


def get_step_explicit_P(K, y, q, lambd, alpha, offset, P_inv):
    """
    Computes the step size for one step of the MM algorithm.
    See Wang and Zou, 2017 for details.
    """
    n_samples = K.shape[0]

    z = y * V_grad(y * (K.dot(alpha) + offset), q=q) / n_samples

    gamma = K.dot(z) + 2 * lambd * K.dot(alpha)
    gamma = np.insert(gamma, 0, z.sum())

    step = (n_samples * q / (q + 1) ** 2) * P_inv @ gamma

    offset_step = step[0]
    alpha_step = step[1:]

    return alpha_step, offset_step


def kern_dwd_obj(K, y, q, lambd, alpha, offset):
    """
    Objective function for kernel DWD.
    """
    return np.mean(V(y * (K.dot(alpha) + offset), q=q) +
                   lambd * alpha.T.dot(K.dot(alpha)))


def c_from_lambd(K, lambd, q, alpha):
    """
    Gets the tuning paramter, C, for the SOCP formulation of DWD
    from lambda.
    """
    beta_norm = np.sqrt(alpha.T.dot(K.dot(alpha)))
    return ((q + 1) ** (q + 1) / q ** q) * beta_norm ** (q + 1)
