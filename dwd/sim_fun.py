import numpy as np
import matplotlib.pyplot as plt


def tuning_curve(X_train, y_train, X_test, y_test,
                 tuning_params, algo, solver_kws={}):
    """
    Plots tuning curve for a provided set of tuning parameters.
    """

    n_tune_values = len(tuning_params)

    train_error = np.zeros(n_tune_values)
    test_error = np.zeros(n_tune_values)

    for i, param in enumerate(tuning_params):
        beta, offset, problem = algo(X_train, y_train, param, solver_kws)

        tr_pred = lin_clf_predict(X_train, beta, offset)
        test_pred = lin_clf_predict(X_test, beta, offset)

        train_error[i] = np.mean(tr_pred != y_train)
        test_error[i] = np.mean(test_pred != y_test)

    plt.plot(tuning_params, train_error, label="Train error")
    plt.plot(tuning_params, test_error, label="Test error")
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.ylim(0, 1)

    return train_error, test_error


def lin_clf_predict(x, beta, offset):
    """
    Returns the linear classifier predictions for a point x.

    sign(x^T beta + offset)

    Parameters
    ----------
    x: array-like, (n_features, )
        The data point.

    beta: array-like, (n_features, )
        The classifier normal vector.

    offset: float
        The classifier offset.
    """
    return np.sign(x.dot(beta) + offset).reshape(-1)


def sample_meatballs(n_samples=1000, n_features=20, n_test=1000,
                     sep=2, p_pos=.5):
    """
    Samples binary classification problem where classes are two spherical
    Gaussians.

    Parameters
    ----------
    n_sample: int
        Sample size of training data.

    n_features: int
        Number of features.

    n_test: int
        Number of test points.

    sep: flot
        Norm of mean difference vector.

    p_pos: float
        Proportion of points in positive class

    Output
    ------
    X_train, y_train, X_test, y_test, beta_true

    """

    mean_neg = np.zeros(n_features)
    mean_pos = sep * np.ones(n_features) / np.sqrt(n_features)

    beta_true = mean_pos - mean_neg

    def sample_xy(n):
        n_pos = int(n * p_pos)
        n_neg = n - n_pos

        X_pos = np.random.multivariate_normal(mean=mean_pos, cov=np.eye(n_features), size=n_pos)
        X_neg = np.random.multivariate_normal(mean=mean_neg, cov=np.eye(n_features), size=n_neg)

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_pos), -1 * np.ones(n_neg)])

        return X, y

    X_train, y_train = sample_xy(n_samples)
    X_test, y_test = sample_xy(n_test)

    return X_train, y_train, X_test, y_test, beta_true


def cv_tuning(gscv, param_name, kind='test', log=True, std=True, color='red'):
    """
    Plots the tuning curve using GridSearchCV.

    Parameters
    ----------
    gscg: from sklearn.model_selection import GridSearchCV
    """
    param_vals = [p[param_name] for p in gscv.cv_results_['params']]

    assert kind in ['train', 'test']
    st = '_{}_score'.format(kind)
    plt.plot(param_vals, gscv.cv_results_['mean' + st],
             color=color, label='cv {}'.format(kind))

    if std:
        plt.plot(param_vals,
                 gscv.cv_results_['mean' + st] + gscv.cv_results_['std' + st],
                 'b--', color=color)
        plt.plot(param_vals,
                 gscv.cv_results_['mean' + st] - gscv.cv_results_['std' + st],
                 'b--', color=color)
    if log:
        plt.xscale('log')
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel(param_name)
