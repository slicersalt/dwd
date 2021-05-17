from abc import ABCMeta
import typing

from sklearn.base import ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot

import numpy as np

__all__ = ['LinearClassifierMixin']


class LinearClassifierMixin(ClassifierMixin):
    """
    Simplified version of sklearn.linear_model._base.LinearClassifierMixin.
    It was removed from the public API, so must be implemented here. Because this is only
    used for DWD and SVM predictors, I do not implement all the extra functionality as in
    the private sklearn.linear_model API.

    Expects instance variables `coef_`, `intercept_`, and `classes_` to be present on the object.
    """

    def decision_function(self, X):
        check_is_fitted(self)

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)

        indices = (scores > 0).astype(int)

        return self.classes_[indices]
