# Updates

## 1.0.5 (2022-01-10)

- Since `cvxpy` is not supported on all platforms, make this an optional dependency installable via `pip install dwd[socp]`. 
- Remove the import aliases from `dwd.__init__`; one must explictly import the solver to be used:
  - `dwd.socp_dwd.DWD` (only in `dwd[socp]`)
  - `dwd.gen_dwd.GenDWD`
  - `dwd.gen_kern_dwd.KernGDWD`
- Add `socp` requirements to readthedocs config so `dwd.socp_dwd` autodoc will generate.

## 1.0.4 (2021-11-19)

- Rolled back dependency version pinning to restore compatibility with other versions of Python.
- Remove matplotlib main dependency

## 1.0.3 (2021-11-19)

- Add sphinx documentation for Read the Docs
- Fix pyproject.toml to match [PEP 621](https://www.python.org/dev/peps/pep-0621/)
- Add property dwd.direction
- Pin dependency versions to support `pip install --require-hashes`

## 1.0.2 (2021-05-28)

- Convert `README.rst` to markdown to be more consistent other documentation.
- Reverted changes to `solve_dwd_socp` to make it [DPP-compliant](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming), as it caused DWD to stall in new versions of cvxpy.

## 1.0.1 (2021-05-17)

- Fix warnings and errors regarding deprecation of [sklearn private API](https://scikit-learn.org/stable/whats_new/v0.22.html#clear-definition-of-the-public-api) so that sklearn 0.22.x and higher are supported
- Fix warnings from cvxpy that `solve_dwd_socp` was not [DPP-compliant](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)

## 1.0.0 (2019-08-17)

- First release!
- Published to PyPI on 2021.05.17 
