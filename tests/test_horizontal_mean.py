import numpy as np
from mom6_tools.drift import HorizontalMeanDiff_da, HorizontalMeanRmse_da
import xarray as xr

# starts with simple cases then build complexity.

# Horizontal mean difference
def test_case1():
  # case 1: 2D, area = none and one cell is masked (land)
  tmp = np.zeros((2,2))
  tmp[0,0] = 1 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanDiff_da(var)
  np.testing.assert_almost_equal(mean, 1./3., decimal=14)

def test_case2():
  # case 2: 2D, area is given (ones) and one cell is masked (land)
  tmp = np.zeros((2,2))
  tmp[0,0] = 1 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[0,-1] = np.nan
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanDiff_da(var, weights=area)
  np.testing.assert_almost_equal(mean, 1./3., decimal=14)

def test_case3():
  # case 3: 2D, area is given (different for each cell) and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[0,-1] = np.nan; tmp[0,0] = 2
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanDiff_da(var, weights=area)
  np.testing.assert_almost_equal(mean, 3./2., decimal=14)

def test_case4():
  # case 4: 2D, area is given (different for each cell) and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[0,-1] = np.nan; tmp[0,0] = 2; tmp[-1,1] = 0.0
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  mean = HorizontalMeanDiff_da(var, weights=area)
  np.testing.assert_almost_equal(mean, 5./3., decimal=14)

def test_case5():
  # case 5: 3D, area = none and one cell is masked (land)
  tmp = np.ones((3,2,2))
  tmp[:,0,0] = 2; tmp[:,0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(3), np.arange(2), np.arange(2)], dims=['z_l','yh', 'xh'])
  tmp = np.ones((3,2,2))
  tmp[:,0,-1] = np.nan; tmp[:,0,0] = 2; tmp[:,-1,1] = 0.0
  area = xr.DataArray(tmp, coords=[np.arange(3), np.arange(2), np.arange(2)], dims=['z_l','yh', 'xh'])
  mean = HorizontalMeanDiff_da(var, weights=area)
  np.testing.assert_almost_equal(mean, np.ones(3)*5./3., decimal=14)

def test_case6():
  # case 6: 4D, area is given (different for each cell) and region (Global) is provided.
  tmp = np.ones((1,3,2,2))
  tmp[:,:,0,0] = 2; tmp[:,:,0,-1] = np.nan
  tmp[:,1::,1,1] = np.nan; tmp[:,2,1,0] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(1), np.arange(3), np.arange(2), np.arange(2)], dims=['time','z_l','yh', 'xh'])
  tmp = np.ones((3,2,2))
  tmp[:,0,-1] = np.nan; tmp[:,0,0] = 2; tmp[:,-1,1] = 0.0
  area = xr.DataArray(tmp, coords=[np.arange(3), np.arange(2), np.arange(2)], dims=['z_l', 'yh', 'xh'])
  tmp = np.ones((1,2,2))
  tmp[:,0,-1] = 0.0
  region = xr.DataArray(tmp, dims=('region', 'yh', 'xh'), coords={'region':['Global']})
  mean = HorizontalMeanDiff_da(var, weights=area, basins=region)
  np.testing.assert_almost_equal(mean, [[[5./3., 5./3.,4./3.]]], decimal=14)

def test_case7():
  # case 7: 4D, area is given (different for each cell) and region (regional, one cell) is provided.
  tmp = np.ones((1,3,2,2))*np.nan
  tmp[:,0,0,0] = 2; tmp[:,1,0,0] = 3; tmp[:,2,0,0] = 4
  var = xr.DataArray(tmp, coords=[np.arange(1), np.arange(3), np.arange(2), np.arange(2)], dims=['time','z_l','yh', 'xh'])
  tmp = np.ones((3,2,2))*np.nan
  tmp[:,0,0] = 2
  area = xr.DataArray(tmp, coords=[np.arange(3), np.arange(2), np.arange(2)], dims=['z_l', 'yh', 'xh'])
  tmp = np.zeros((1,2,2))
  tmp[:,0,0] = 1.0
  region = xr.DataArray(tmp, dims=('region', 'yh', 'xh'), coords={'region':['Regional']})
  mean = HorizontalMeanDiff_da(var, weights=area, basins=region)
  np.testing.assert_almost_equal(mean, [[[2., 3., 4.]]], decimal=14)

# Horizontal mean RMS
def test_case8():
  # case 8: area = none and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  rms = HorizontalMeanRmse_da(var)
  np.testing.assert_almost_equal(rms, np.sqrt(2.), decimal=14)

def test_case9():
  # case 9: area is given (ones) and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[0,-1] = np.nan
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  rms = HorizontalMeanRmse_da(var, weights=area)
  np.testing.assert_almost_equal(rms, np.sqrt(2.), decimal=14)

def test_case10():
  # case 10: area is given (different for each cell) and one cell is masked (land)
  tmp = np.ones((2,2))
  tmp[0,0] = 2 ; tmp[0,-1] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  tmp = np.ones((2,2))
  tmp[1,0] = 2 ; tmp[0,-1] = np.nan
  area = xr.DataArray(tmp, coords=[np.arange(2), np.arange(2)], dims=['yh', 'xh'])
  rms = HorizontalMeanRmse_da(var, weights=area)
  np.testing.assert_almost_equal(rms, np.sqrt(7./4.), decimal=14)

def test_case11():
  # case 11: 4D, area is given (different for each cell) and region (Global) is provided.
  tmp = np.ones((1,3,2,2))
  tmp[:,:,0,0] = 2; tmp[:,:,0,-1] = np.nan
  tmp[:,1::,1,1] = np.nan; tmp[:,2,1,0] = np.nan
  var = xr.DataArray(tmp, coords=[np.arange(1), np.arange(3), np.arange(2), np.arange(2)], dims=['time','z_l','yh', 'xh'])
  tmp = np.ones((3,2,2))
  tmp[:,0,-1] = np.nan; tmp[:,0,0] = 2; tmp[:,-1,1] = 0.0
  area = xr.DataArray(tmp, coords=[np.arange(3), np.arange(2), np.arange(2)], dims=['z_l', 'yh', 'xh'])
  tmp = np.ones((1,2,2))
  tmp[:,0,-1] = 0.0
  region = xr.DataArray(tmp, dims=('region', 'yh', 'xh'), coords={'region':['Global']})
  rms = HorizontalMeanRmse_da(var, weights=area, basins=region)
  np.testing.assert_almost_equal(rms, [[[np.sqrt(3.), np.sqrt(3.), np.sqrt(8./3.)]]], decimal=14)

def test_case12():
  # case 12: 4D, area is given (different for each cell) and region (regional, one cell) is provided.
  tmp = np.ones((1,3,2,2))*np.nan
  tmp[:,0,0,0] = 2; tmp[:,1,0,0] = 3; tmp[:,2,0,0] = 4
  var = xr.DataArray(tmp, coords=[np.arange(1), np.arange(3), np.arange(2), np.arange(2)], dims=['time','z_l','yh', 'xh'])
  tmp = np.ones((3,2,2))*np.nan
  tmp[:,0,0] = 2
  area = xr.DataArray(tmp, coords=[np.arange(3), np.arange(2), np.arange(2)], dims=['z_l', 'yh', 'xh'])
  tmp = np.zeros((1,2,2))
  tmp[:,0,0] = 1.0
  region = xr.DataArray(tmp, dims=('region', 'yh', 'xh'), coords={'region':['Regional']})
  rms = HorizontalMeanRmse_da(var, weights=area, basins=region)
  np.testing.assert_almost_equal(rms, [[[2., 3., 4.]]], decimal=14)
