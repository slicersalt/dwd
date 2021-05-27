# Updates

- Convert `README.rst` to markdown to be more consistent other documentation.
- Reverted changes to `solve_dwd_socp` to make it [DPP-compliant](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming), as it caused DWD to stall in new versions of cvxpy.

## 2021.05.17 (version 1.0.1)

- Fix warnings and errors regarding deprecation of [sklearn private API](https://scikit-learn.org/stable/whats_new/v0.22.html#clear-definition-of-the-public-api) so that sklearn 0.22.x and higher are supported
- Fix warnings from cvxpy that `solve_dwd_socp` was not [DPP-compliant](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)

## 2019.08.17 (version 1.0.0)

- First release!
- Published to PyPI on 2021.05.17 
