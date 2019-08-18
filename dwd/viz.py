import matplotlib.pyplot as plt
import numpy as np


class ABLine2D(plt.Line2D):
    """
    Draw a line based on its slope and y-intercept. Additional arguments are
    passed to the <matplotlib.lines.Line2D> constructor.

    from https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
    """

    def __init__(self, slope, intercept, *args, **kwargs):

        # get current axes if user has not specified them
        if'axes' not in kwargs.keys():
            kwargs.update({'axes': plt.gca()})
        ax = kwargs['axes']

#         # if unspecified, get the current line color from the axes
#         if not ('color' in kwargs or 'c' in kwargs):
#             kwargs.update({'color':ax._get_lines.color_cycle.next()})

        # init the line, add it to the axes
        super(ABLine2D, self).__init__([], [], *args, **kwargs)
        self._slope = slope
        self._intercept = intercept
        ax.add_line(self)

        # cache the renderer, draw the line for the first time
        ax.figure.canvas.draw()
        self._update_lim(None)

        # connect to axis callbacks
        self.axes.callbacks.connect('xlim_changed', self._update_lim)
        self.axes.callbacks.connect('ylim_changed', self._update_lim)

    def _update_lim(self, event):
        """ called whenever axis x/y limits change """
        x = np.array(self.axes.get_xbound())
        y = (self._slope * x) + self._intercept
        self.set_data(x, y)
        self.axes.draw_artist(self)


def clf2D_slope_intercept(coef=None, intercept=None, clf=None):
    """
    Gets the slop an intercept for the separating hyperplane of a linear
    classifier fit on a two dimensional dataset.

    Parameters
    ----------
    coef:
        The classification normal vector.

    intercept:
        The classifier intercept.

    clf: subclass of sklearn.linear_model.base.LinearClassifierMixin
        A sklearn classifier with attributes coef_ and intercept_

    Output
    ------
    slope, intercept
    """

    if clf is not None:
        coef = clf.coef_.reshape(-1)
        intercept = float(clf.intercept_)
    else:
        assert coef is not None and intercept is not None

    slope = - coef[0] / coef[1]
    intercept = - intercept / coef[1]

    return slope, intercept


def get_mesh_grid(X, mesh_step=.2, excess=.5):
    """
    Returns a 2D mesh grid useful.
    """

    x_min, x_max = X[:, 0].min() - excess, X[:, 0].max() + excess
    y_min, y_max = X[:, 1].min() - excess, X[:, 1].max() + excess
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    return X_mesh, xx, yy
