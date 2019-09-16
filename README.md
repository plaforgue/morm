# Medians of (Randomized) Means and U-statistics

## Summary

This Python package contains code to compute Medians of (Randomized) Means and U-statistics. The main functions are `MoM` (for Median of Means), `MoRM` (for Median of Randomized Means), `MoCU` (for Median of Complete U-statistics) and `MoIU` (for Median of Incomplete U-statistics). They take as input a 2-dimensional array `X`, and hyperparameters such as the number of blocks `K`, the size of the blocks `B`, and possibly the sampling scheme or the kernel for U-statistics. They return the corresponding estimator.


## Installation
To install the package, simply clone it, and then do:

  `$ pip install -e .`

To check that everything worked, the command

  `$ python -c 'import morm'`

should not return any error.


## Use
Here is a toy example (complete version is available at `examples/toy_example.py`)
```python
import numpy as np
from morm import MoM, MoRM, MoCU, MoIU

mean, var = 0.0, 1.0
n = 1000
X = np.random.normal(loc=mean, scale=var, size=n).reshape(-1, 1)

K = 10
B = 100

# Mean estimation
mom = MoM(X, K)
morm = MoRM(X, K, B)
print("mean : %s" % mean)
print("mom  : %s" % mom)
print("morm : %s" % morm)

print("\n")

# Variance estimation
mocu = MoCU(X, K, kernel='squared_norm')
moiu = MoIU(X, K, B, kernel='squared_norm')
print("var  : %s" % var)
print("mocu : %s" % mocu)
print("moiu : %s" % moiu)
```
