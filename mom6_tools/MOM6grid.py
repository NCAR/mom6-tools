import xarray as xr

# -- return MOM6 grid object
def MOM6grid(grd_file):
    """
    Return an object with MOM6 grid data
    MOM6_hgrd = MOM6grid(path-to-static_file)
    """

    # create an empty class object
    class MOM6_grd:
        pass

    # open grid file
    try: nc = xr.open_dataset(grd_file, decode_times=False)
    except: raise Exception('Could not find file', grd_file)

    # fill grid object
    for var in nc.variables:
       #dummy = str("MOM6_grd.%s = nc.variables[var][:]"% (var))
       dummy = str("MOM6_grd.%s = nc.%s[:].values"% (var,var))
       exec(dummy)

    # close netcdf file
    nc.close()
    print('MOM6 grid successfully loaded... \n')
    return MOM6_grd

