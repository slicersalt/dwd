import numpy as np


def pm1(y):
    """
    Converts binary label vector y into +/- 1s

    Parameters
    ----------
    y: array-like, (n_samples, )
        The original labels. Must have binary labels.

    Output
    ------
    y: array-like, (n_samples, )
        y with +/- 1 values. Note the positive label is determined
        by np.unique(y)[1].
    """
    y = np.array(y).reshape(-1)
    labels = np.unique(y)

    # check binary
    if len(labels) != 2:
        raise ValueError('y must have binary labels;'
                         ' found {} labels'.format(len(labels)))

    y_pm1 = np.ones(len(y))
    y_pm1[y == labels[0]] = -1
    return y_pm1
