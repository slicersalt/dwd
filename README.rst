DWD
----

Overview
========

This package implements Distance Weighted Discrimination (DWD). DWD For details see
(`Marron et al 2007`_, `Wang and Zou 2018`_). Originally implemented in Python by `Iain
Carmichael`_. Currently maintained by `Kitware, Inc`_.

The package currently implements:

- Original DWD formulation solved with Second Order Cone Programming (SOCP) and solved using cvxpy.

- Genralized DWD (gDWD) and kernel gDWD solved with the Majorization-Minimization algorithm presented in Wang and Zou, 2018.


Marron, James Stephen, Michael J. Todd, and Jeongyoun Ahn. "Distance-weighted discrimination." Journal of the American Statistical Association 102, no. 480 (2007): 1267-1271.

Wang, Boxiang, and Hui Zou. "Another look at distance‐weighted discrimination." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 80, no. 1 (2018): 177-198.

Installation
============

The DWD package can be installed via pip or github. This package is currently only tested in python 3.6.

::

    pip install dwd


::

    git clone https://github.com/idc9/dwd.git
    python setup.py install

Example
=======

.. code:: python

    from sklearn.datasets import make_blobs, make_circles
    from dwd import DWD, KernGDWD

    # sample sythetic training data
    X, y = make_blobs(n_samples=200, n_features=2,
                      centers=[[0, 0],
                               [2, 2]])

    # fit DWD classifier
    dwd = DWD(C='auto').fit(X, y)

    # compute training accuracy
    dwd.score(X, y)

    0.94

.. image:: https://raw.githubusercontent.com/slicersalt/dwd/master/doc/figures/dwd_sep_hyperplane.png

.. code:: python

    # sample some non-linear, toy data
    X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)

    # fit kernel DWD wit gaussian kernel
    kdwd = KernGDWD(lambd=.1, kernel='rbf',
                    kernel_kws={'gamma': 1}).fit(X, y)

    # compute training accuracy
    kdwd.score(X, y)

    0.915

.. image:: https://raw.githubusercontent.com/slicersalt/dwd/master/doc/figures/kern_dwd.png

For more example code see `these example notebooks`_ (including the code to generate the above figures). If the notebooks aren't loading on github you can copy/paste the notebook url into https://nbviewer.jupyter.org/.

Help and Support
================

Additional documentation, examples and code revisions are coming soon.

Documentation
^^^^^^^^^^^^^

The source code is located on github: https://github.com/slicersalt/dwd

Testing
^^^^^^^

Testing is done using `nose`.

Contributing
^^^^^^^^^^^^

We welcome contributions to make this a stronger package: data examples,
bug fixes, spelling errors, new features, etc.

.. _Iain Carmichael: https://idc9.github.io/
.. _Marron et al 2007: https://amstat.tandfonline.com/doi/abs/10.1198/016214507000001120?casa_token=9u7plrafGzkAAAAA:10_e1f_4dQmNusX2G_YsXgKCuhQWUG2CyKqOtq0Ukev092euOhQ7p51i44B1ZbMeOKI4FvUJl2bjYQ
.. _Wang and Zou 2018: https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12244
.. _these example notebooks: https://github.com/idc9/dwd/tree/master/doc/example_notebooks
.. _Kitware, Inc: https://kitware.com/
