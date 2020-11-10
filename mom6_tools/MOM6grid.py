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
    # fixes non-monotonic longitudes
    mgeolon = xr.where(nc.geolon < nc.geolon[-1,0],nc.geolon+360,nc.geolon).rename({'mgeolon'})
    mgeolon_u = xr.where(nc.geolon_u < nc.geolon_u[-1,0],nc.geolon_u+360.0,nc.geolon_u).rename({'mgeolon_u'})
    mgeolon_c = mgeolon_u.rename({'mgeolon_c'})
    mgeolon_v = mgeolon.rename({'mgeolon_v'})

    if xrformat:
      nc['mgeolon']   = mgeolon
      nc['mgeolon_c'] = mgeolon_c
      nc['mgeolon_u'] = mgeolon_u
      nc['mgeolon_v'] = mgeolon_v
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

      MOM6_grd.mgeolon   = mgeolon.values
      MOM6_grd.mgeolon_c = mgeolon_c.values
      MOM6_grd.mgeolon_u = mgeolon_u.values
      MOM6_grd.mgeolon_v = mgeolon_v.values
      # close netcdf file
      nc.close()
      print('MOM6 grid successfully loaded... \n')
      return MOM6_grd

