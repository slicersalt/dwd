import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KernelCenterer

from dwd.kernel_utils import KernelClfMixin, KernelScaler


class KernMD(BaseEstimator, KernelClfMixin):
    def __init__(self, kernel='linear', kernel_kws={}, naive_bayes=False):
        self.kernel = kernel
        self.kernel_kws = kernel_kws

        self.naive_bayes = naive_bayes

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._Xfit = X  # Store K so we can compute predictions

        K = self._compute_kernel(X)

        self.dual_coef_, self.intercept_ = \
            kern_md(K, y, naive_bayes=self.naive_bayes)

        self.intercept_ = self.intercept_.reshape(-1)
        self.dual_coef_ = self.dual_coef_.reshape(1, -1)

        return self


def kern_md(K, y, naive_bayes=False):
    """
    Returns the coefficients for the kernel mean difference
    classifier.

    Parameters
    ----------
    K: array-like (n_samples, n_samples)
        Kernel matrix

    y: array-like (n_samples, )
        Vector of binary labels.

    naive_bayes: bool
        Compute naive bayes direction.

    Output
    ------
    alpha, intercept

    """
    labels = np.unique(y)
    assert K.shape[0] == K.shape[1]
    assert len(labels) == 2  # make sure binary classifier
    assert len(y) == K.shape[0]

    # center and scale K to compute Naive Bayes
    # TODO: check intercept
    if naive_bayes:
        K = KernelCenterer().fit_transform(K)
        K = KernelScaler().fit_transform(K)

    pos_ind = y == labels[1]
    neg_ind = y == labels[0]
    n_pos = sum(pos_ind)
    n_neg = sum(neg_ind)

    y_tilde = (pos_ind / n_pos) - (neg_ind / n_neg)
    alpha = K.dot(y_tilde)

    intercept = (1.0 / n_pos ** 2) * pos_ind.T.dot(K.dot(pos_ind)) - \
        (1.0 / n_neg ** 2) * neg_ind.T.dot(K.dot(neg_ind))

    return alpha, intercept
