# Overview

This package implements Distance Weighted Discrimination (DWD). DWD For details see
([Marron et al 2007][marron-et-al], [Wang and Zou 2018][wang-zou]). Originally
implemented in Python by [Iain Carmichael][iain-carmichael]. Currently maintained by
[Kitware, Inc][kitware].

The package currently implements:

- Original DWD formulation solved with Second Order Cone Programming (SOCP) and solved
using cvxpy.

- Genralized DWD (gDWD) and kernel gDWD solved with the Majorization-Minimization
algorithm presented in Wang and Zou, 2018.


Marron, James Stephen, Michael J. Todd, and Jeongyoun Ahn. "Distance-weighted
discrimination." Journal of the American Statistical Association 102, no. 480 (2007):
1267-1271.

Wang, Boxiang, and Hui Zou. "Another look at distance‐weighted discrimination." Journal
of the Royal Statistical Society: Series B (Statistical Methodology) 80, no. 1 (2018):
177-198.

# Installation

The DWD package can be installed via pip or github. This package is currently only
tested in python 3.6.

```
$ pip install dwd
```

The conic solver `socp_dwd.DWD` depends on `cvxpy`, which is not available on all platforms. See [the `cvxpy` installation instructions][cvxpy]. If `cvxpy` dependencies are met, then use `pip install dwd[socp]`. 

[Flit][flit] is used for packaging, and all package metadata is stored in `pyproject.toml`. To install this project locally or for development, use `flit install` or build a pip-installable wheel with `flit build`.

[cvxpy]: https://www.cvxpy.org/install/index.html
[flit]: https://github.com/takluyver/flit

# Example

```python
from sklearn.datasets import make_blobs
from dwd.socp_dwd import DWD

# sample sythetic training data
X, y = make_blobs(
    n_samples=200,
    n_features=2,
    centers=[[0, 0],
             [2, 2]],
)

# fit DWD classifier
dwd = DWD(C='auto').fit(X, y)

# compute training accuracy
dwd.score(X, y)  # 0.94
```

![dwd_sep_hyperplane][dwd_sep_hyperplane]

```python
from sklearn.datasets import make_circles
from dwd.gen_kern_dwd import KernGDWD

# sample some non-linear, toy data
X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)

# fit kernel DWD wit gaussian kernel
kdwd = KernGDWD(
    lambd=.1, kernel='rbf',
    kernel_kws={'gamma': 1},
).fit(X, y)

# compute training accuracy
kdwd.score(X, y)  # 0.915
```

![kern_dwd][kern_dwd]

For more example code see [these example notebooks][example-notebooks] (including the code
to generate the above figures). If the notebooks aren't loading on github you can copy/paste the notebook url into https://nbviewer.jupyter.org/.

# Help and Support

Additional documentation, examples and code revisions are coming soon.

## Documentation

The source code is located on github: https://github.com/slicersalt/dwd

## Testing

Testing is done using `nose`.

## Contributing

We welcome contributions to make this a stronger package: data examples,
bug fixes, spelling errors, new features, etc.

[iain-carmichael]: https://idc9.github.io/
[kitware]: https://kitware.com/

[marron-et-al]: https://amstat.tandfonline.com/doi/abs/10.1198/016214507000001120
[wang-zou]: https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12244

[dwd_sep_hyperplane]: https://raw.githubusercontent.com/slicersalt/dwd/master/doc/figures/dwd_sep_hyperplane.png
[kern_dwd]: https://raw.githubusercontent.com/slicersalt/dwd/master/doc/figures/kern_dwd.png
[example-notebooks]: https://github.com/idc9/dwd/tree/master/doc/example_notebooks
