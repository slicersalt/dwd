import cvxpy as cp
import numpy as np

from sklearn.base import BaseEstimator  # , TransformerMixin, ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin


def solve_svm(X, y, C, sample_weight=None, solver_kws={}):
    """
    Solves soft-margin SVM problem.

    min_{beta, intercept}
    (1/n) * sum_{i=1}^n [1  - y_i * (x^T beta + intercept)] + C * ||beta||_1

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
    beta, intercept, problem

    beta: (n_features, )
        SVM normal vector.

    intercept: float
        SVM intercept.

    problem: cp.Problem

    y_hat = np.sign(x.dot(beta) + intercept)
    """
    if sample_weight is not None:
        raise NotImplementedError

    n_samples, n_features = X.shape
    y = y.reshape(-1, 1)

    beta = cp.Variable((n_features, 1))
    intercept = cp.Variable()
    C = cp.Parameter(value=C, nonneg=True)

    # TODO: should we make this + intercept
    loss = cp.sum(cp.pos(1 - cp.multiply(y, X * beta + intercept)))
    reg = cp.norm(beta, 1)
    objective = loss / n_samples + C * reg

    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(**solver_kws)

    return beta.value, intercept.value, problem


class SVM(BaseEstimator, LinearClassifierMixin):

    def __init__(self, C=1.0, solver_kws={}):
        self.C = C
        self.solver_kws = solver_kws

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)

        self.coef_, self.intercept_, self.problem_ = \
            solve_svm(X, y, C=self.C, sample_weight=sample_weight,
                      solver_kws=self.solver_kws)

        self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = self.intercept_.reshape(-1)

        return self
