import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import ClassifierMixin


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
