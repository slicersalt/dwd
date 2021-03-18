from setuptools import setup, find_packages

install_requires = ['numpy', 'cvxpy', 'sklearn', 'matplotlib']

setup(
      name='dwd',
      version='1.0.1',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      maintainer='Kitware Medical',
      maintainer_email='david.allemang@kitware.com',
      license='MIT',
      description='Distance weighted discrimination for Python',
      long_description=(
            'This package implements Distance Weighted Discrimination (DWD). DWD For '
            'details see (Marron et al 2007, Wang and Zou 2018). Originally implemented '
            'in Python by Iain Carmichael.'
      ),
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False
)
