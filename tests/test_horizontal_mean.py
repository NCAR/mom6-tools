import numpy as np
from mom6_tools.HorizontalMean import HorizontalMeanDiff_da, HorizontalMeanRmse_da
import xarray as xr

def test_case1():
  # case 1: area = none and one cell is masked (land)
  tmp = np.zeros((2,2))
  tmp[0,0] = 1 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanDiff_da(var)
  np.testing.assert_almost_equal(mean, 1./3., decimal=14)

def test_case2():
  # case 2: area is given (ones) and one cell is masked (land)
  tmp = np.zeros((2,2))
  tmp[0,0] = 1 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[0,-1] = np.nan
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanDiff_da(var, weights=area)
  np.testing.assert_almost_equal(mean, 1./3., decimal=14)

def test_case3():
  # case 3: area is given (different for each cell) and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[0,-1] = np.nan; tmp[0,0] = 2
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanDiff_da(var, weights=area)
  np.testing.assert_almost_equal(mean, 3./2., decimal=14)

def test_case4():
  # case 4: area is given (different for each cell) and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[0,-1] = np.nan; tmp[0,0] = 2; tmp[-1,1] = 0.0
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanDiff_da(var, weights=area)
  np.testing.assert_almost_equal(mean, 5./3., decimal=14)

# RMSE tests

def test_case5():
  # case 5: area = none and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanRmse_da(var)
  np.testing.assert_almost_equal(mean, np.sqrt(2.), decimal=14)

def test_case6():
  # case 6: area is given (ones) and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[0,-1] = np.nan
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanRmse_da(var, weights=area)
  np.testing.assert_almost_equal(mean, np.sqrt(2.), decimal=14)

def test_case7():
  # case 6: area is given (different for each cell) and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[1,0] = 2 ; tmp[0,-1] = np.nan
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanRmse_da(var, weights=area, debug=True)
  np.testing.assert_almost_equal(mean, np.sqrt(7./4.), decimal=14)
