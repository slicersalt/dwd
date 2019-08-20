import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class KernelClfMixin(ClassifierMixin):
    """
    Mixin for kernel classifiers.
    """
    # def __init__(self, kernel, kernel_kws={}):
    #     # TODO: kernel centering
    #     self.kernel = kernel
    #     self.kernel_kws = kernel_kws

    def fit(self, X, y, sample_weights=None):

        # the fit function needs to set the following
        self.classes_ = np.unique(y)
        self.__Xfit = X
        self.intercept_ = None
        self.dual_coef_ = None
        raise NotImplementedError

    def _compute_kernel(self, X):
        """

        Parameters
        ----------
        X: array-like, shape (n_samples_test, n_features)
            A matrix of new data

        Returns
        -------
        K: array-like, shape (n_samples_train, n_samples_test)
        """

        if self.kernel == 'precomputed':
            return X

        elif callable(self.kernel):
            return self.kernel(X=self._Xfit,
                               Y=X,
                               kernel=self.kerel)

        elif isinstance(self.kernel, str):
            return pairwise_kernels(X=self._Xfit,
                                    Y=X,
                                    metric=self.kernel,
                                    **self.kernel_kws)

    def decision_function(self, X):
        """Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """

        K = self._compute_kernel(X)

        scores = safe_sparse_dot(K.T, self.dual_coef_.T,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]


class KernelScaler(BaseEstimator, TransformerMixin):
    """Center a kernel matrix
    Let K(x, z) be a kernel defined by phi(x)^T phi(z), where phi is a
    function mapping x to a Hilbert space. KernelScaler scales (i.e.,
    normalized to have zero norm) the data without explicitly computing phi(x).
    It is equivalent to centering phi(x) with
    sklearn.preprocessing.StandardScaler(with_mean=False).
    Read more in the :ref:`User Guide <kernel_centering>`.
    """

    def __init__(self):
        # Needed for backported inspect.signature compatibility with PyPy
        pass

    def fit(self, K, y=None):
        """Fit KernelCenterer
        Parameters
        ----------
        K : numpy array of shape [n_samples, n_samples]
            Kernel matrix.
        Returns
        -------
        self : returns an instance of self.
        """
        K = check_array(K, dtype=FLOAT_DTYPES)
        self.K_diag_ = np.diag(K)
        return self

    def transform(self, K, copy=True):
        """Center kernel matrix.
        Parameters
        ----------
        K : numpy array of shape [n_samples1, n_samples2]
            Kernel matrix.
        copy : boolean, optional, default True
            Set to False to perform inplace computation.
        Returns
        -------
        K_new : numpy array of shape [n_samples1, n_samples2]
        """
        # check_is_fitted(self, 'K_diag_')

        K = check_array(K, copy=copy, dtype=FLOAT_DTYPES)

        n = len(self.K_diag_)
        s = 1.0 / np.sqrt(self.K_diag_ / n)

        return np.multiply(np.multiply(s, K), s)

    @property
    def _pairwise(self):
        return True
