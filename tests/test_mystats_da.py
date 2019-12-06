import numpy as np
from mom6_tools.stats import myStats_da
import xarray as xr

# starts with simple cases then build complexity.

# Calculates min, max, mean, standard deviation and root-mean-square for DataArray ds

# starts with simple cases then build complexity.
# First, basins = None
def test_case1():
  # case 1, weight=1, var=1 and only one cell is masked
  tmp = np.ones((1,2,2))
  tmp[0,0,1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(1), np.arange(2), np.arange(2)], \
                   dims=['time', 'yh', 'xh'])
  weight = xr.DataArray(tmp[0,:], coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  stats = myStats_da(var, weight)
  # order is min, max, mean, std, rms
  answers = [1., 1., 1., 0., 1.]
  for i in range(len(answers)):
    np.testing.assert_almost_equal(stats[0,i,:].values, answers[i], decimal=14)

def test_case2():
  # case 2: weight varies, var=1 and only one cell is masked
  tmp = np.ones((1,2,2))
  tmp[0,0,1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(1), np.arange(2), np.arange(2)], \
                     dims=['time', 'yh', 'xh'])
  tmp2 = tmp.copy()
  tmp2[0,0,0] = 2
  weight = xr.DataArray(tmp2[0,:], coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  stats = myStats_da(var, weight)
  # order is min, max, mean, std, rms
  answers = [1., 1., 1., 0., 1.]
  for i in range(len(answers)):
    np.testing.assert_almost_equal(stats[0,i,:].values, answers[i], decimal=14)

def test_case3():
  # case 3: weights are one, var varies and no one cell is masked
  tmp = np.ones((1,1,5))
  tmp[0,0,:] = np.ones(5) * [-2., 5., -8., 9., -4.]
  var = xr.DataArray(tmp, coords=[np.arange(1), np.arange(1), np.arange(5)], \
                   dims=['time', 'yh', 'xh'])
  tmp1 = np.ones((1,5))
  weight = xr.DataArray(tmp1, coords=[np.arange(1), np.arange(5)], dims=['yh', 'xh'])
  stats = myStats_da(var, weight)
  # order is min, max, mean, std, rms
  answers = [-8, 9., 0., 6.16441400296898, 6.16441400296898]
  for i in range(len(answers)):
    np.testing.assert_almost_equal(stats[0,i,:].values, answers[i], decimal=14)
