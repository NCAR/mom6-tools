import xarray as xr

# -- return MOM6 grid object
def MOM6grid(grd_file, xrformat=False):
    """
    Return an object or xarray Dataset with the MOM6 grid data.

    Parameters
    ----------
    grd_file : str
    Path to the static file.

    xrformat : boolean, optional
    If True, returns an xarray Dataset. Otherwise (default), returns an
    object with numpy arrays.

    Returns
    -------
    """

    # open grid file
    try: nc = xr.open_dataset(grd_file, decode_times=False)
    except: raise Exception('Could not find file', grd_file)

    if xrformat:
      print('MOM6 grid successfully loaded... \n')
      return nc

    else:
      # create an empty class object
      class MOM6_grd:
          pass


      # fill grid object
      for var in nc.variables:
         #dummy = str("MOM6_grd.%s = nc.variables[var][:]"% (var))
         dummy = str("MOM6_grd.%s = nc.%s[:].values"% (var,var))
         exec(dummy)

      # close netcdf file
      nc.close()
      print('MOM6 grid successfully loaded... \n')
      return MOM6_grd

