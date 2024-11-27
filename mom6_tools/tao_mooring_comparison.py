#!/usr/bin/env python

import xarray as xr
import numpy as np
import nc_time_axis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import warnings, os, yaml, argparse
import pandas as pd
import dask, intake
from datetime import datetime, date
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.m6toolbox import weighted_temporal_mean, mom6_latlon2ij, cime_xmlquery

def parseCommandLine():
    """
    Parse the command line positional and optional arguments.
    This is the highest level procedure invoked from the very end of the script.
    """
    parser = argparse.ArgumentParser(description=
      '''
      Compares (model vs obs) T, S and U near the Equator.
      ''',
         epilog='Written by Frank Bryan (bryan@ucar.edu).')
    parser.add_argument('diag_config_yml_path', type=str,
                        help='''Full path to the yaml file describing the run and diagnostics to be performed.''')
    parser.add_argument('-sd','--start_date', type=str, default='',
                        help='''Start year to compute averages. Default is to use value set in diag_config_yml_path''')
    parser.add_argument('-ed','--end_date', type=str, default='',
                        help='''End year to compute averages. Default is to use value set in diag_config_yml_path''')
    parser.add_argument('-nw','--number_of_workers',  type=int, default=0,
                        help='''Number of workers to use (default=0, serial job).''')
    parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''', action="store_true")
    optCmdLineArgs = parser.parse_args()
    driver(optCmdLineArgs)

    #-- This is where all the action happends, i.e., functions for each diagnostic are called.

def driver(args):
    nw = args.number_of_workers
    path_plt_out = 'PNG/TAOMooring/'
    if not os.path.isdir('PNG/TAOMooring'):
        print('Creating a directory to place figures (PNG/TAOMooring)... \n')
        os.system('mkdir -p PNG/TAOMooring')
    if not os.path.isdir('ncfiles'):
        print('Creating a directory to place netCDF files (ncfiles)... \n')
        os.system('mkdir ncfiles')

    # Read in the yaml file
    diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)

    caseroot = diag_config_yml['Case']['CASEROOT']
    args.casename = cime_xmlquery(caseroot, 'CASE')
    DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
    if DOUT_S:
      OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')+'/ocn/hist/'
    else:
      OUTDIR = cime_xmlquery(caseroot, 'RUNDIR')

    # file streams
    args.monthly = args.casename+diag_config_yml['Fnames']['z']
    args.static = args.casename+diag_config_yml['Fnames']['static']
    args.geom = args.casename+diag_config_yml['Fnames']['geom']
    print('Output directory is:', OUTDIR)
    print('Casename is:', args.casename)
    print('Monthly file is:', args.monthly)
    print('Static file is:', args.static)
    print('Number of workers: ', nw)

    # set avg dates
    avg = diag_config_yml['Avg']
    if not args.start_date : args.start_date = avg['start_date']
    if not args.end_date : args.end_date = avg['end_date']


    # read grid info
    geom_file = OUTDIR+'/'+args.geom
    if os.path.exists(geom_file):
      grd = MOM6grid(OUTDIR+'/'+args.static, geom_file, xrformat=True)
    else:
      grd = MOM6grid(OUTDIR+'/'+args.static, xrformat=True)

    # Get index for equator on model grid
    jeq = np.abs(grd['geolat'][:,0]).argmin().values
    print('j_eq = ',jeq,' lat=',grd['geolat'][jeq,0].values)

    # load obs
    path_obs = '/glade/campaign/cgd/oce/datasets/obs/TAO_adcp_mon'

    parallel = False
    if nw > 1:
        parallel = True
        cluster = NCARCluster()
        cluster.scale(nw)
        client = Client(cluster)

    print('Reading monthly dataset ...')
    startTime = datetime.now()

    def preprocess(ds):
        variables = ['uo', 'time', 'z_l', 'z_i']
        return ds[variables]

    # The full case archive
    ds = xr.open_mfdataset(os.path.join(OUTDIR,args.monthly),
                        data_vars='minimal',coords='minimal',compat='override',
                        parallel=parallel,
                        preprocess=preprocess)

    print('Time elapsed: ',datetime.now() - startTime)

    print('Selecting data between {} and {} (time) and yh={})...'.format(args.start_date, \
                                                                     args.end_date,jeq))
    startTime = datetime.now()

    # Subset the selected time period and along the equator upper ocean
    ds_sel = ds.sel(time=slice(args.start_date, args.end_date),z_i=slice(0,500),z_l=slice(0,500)).isel(yh=jeq)
    print('Time elasped: ', datetime.now() - startTime)

    print('Clim. Mean and Annual Cycle then time averaging...')
    startTime = datetime.now()
    uo_ann_clim = weighted_temporal_mean(ds_sel,'uo').mean('time')
    uo_mon_clim = ds_sel['uo'].transpose().groupby('time.month').mean('time').squeeze().compute()
    print('Time elasped: ', datetime.now() - startTime)

    if parallel:
        print('\n Releasing workers...')
        client.close(); cluster.close()

    freq_mooring = 'mon'
    pos_mooring = ['0n147e', '0n156e', '0n165e', '0n170w', '0n140w', '0n110w', # Pacific
                   '0n23w',                                                    # Atlantic
                   '0n67e', '0n80.5e', '0n90e']                                # Indian

    print('Plots along fixed longitudes...')
    ulev = np.arange(-150,151,10)
    cmap = 'coolwarm'
    positions = np.arange(1,13)
    labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    def fmt(x):
        s = f"{x:.0f}"
        return f"{s}"

    for pos in pos_mooring:
        file_adcp = 'adcp' + pos + '_mon.cdf'
        print('Getting ',file_adcp)
        ds_adcp = xr.open_dataset(os.path.join(path_obs,file_adcp))
        ds_adcp['u_adcp_mask'] = ds_adcp['u_1205'].where(ds_adcp['QU_5205'] != 0,np.nan,drop=False).squeeze().transpose()

        lon_tao = ds_adcp['lon']
        iplt_m,jplt_m = mom6_latlon2ij(grd['geolon'],grd['geolat'],lon_tao,0.0)
        print('Mooring ',pos,' lon=',lon_tao.values,' iplt_m,jplt_m=',iplt_m,jplt_m)

        u_adcp_annclim = weighted_temporal_mean(ds_adcp,'u_adcp_mask').mean('time')
        u_adcp_monclim = ds_adcp['u_adcp_mask'].groupby('time.month').mean('time')

        fig,ax=plt.subplots(ncols=3,sharey=True,
                            figsize=(9,4),constrained_layout=True)

        fig.suptitle(pos + ' (' + args.start_date + ' - ' + args.end_date + ')')

        ax[0].contourf(u_adcp_monclim['month'],u_adcp_monclim['depth'],u_adcp_monclim,
                       levels=ulev,cmap=cmap,extend='both')
        clo=ax[0].contour(u_adcp_monclim['month'],u_adcp_monclim['depth'],u_adcp_monclim,
                          levels=ulev,colors='k',linewidths=1)
        ax[0].clabel(clo,clo.levels,inline=True,fmt=fmt,fontsize=10)
        ax[0].set_title('ADCP')
        ax[0].xaxis.set_major_locator(ticker.FixedLocator(positions))
        ax[0].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
        ax[0].set_ylim(400,0)
        ax[0].set_ylabel('Depth (m)')

        month = uo_mon_clim['month']
        z = uo_mon_clim['z_l']
        u = uo_mon_clim.isel(xq=iplt_m)*100
        cf = ax[1].contourf(month,z,u,levels=ulev,cmap=cmap,extend='both')
        cl = ax[1].contour(month,z,u,levels=ulev,colors='k',linewidths=1)
        ax[1].clabel(cl,cl.levels,inline=True,fmt=fmt,fontsize=10)
        ax[1].set_title('MOM6')
        ax[1].xaxis.set_major_locator(ticker.FixedLocator(positions))
        ax[1].xaxis.set_major_formatter(ticker.FixedFormatter(labels))

        cbar = fig.colorbar(cf,ax=ax[0:2],extend='both',location='bottom',shrink=0.9)
        cbar.ax.set_xlabel('$cm s^{-1}$')

        ax[2].plot(u_adcp_annclim,u_adcp_annclim['depth'],color='k',label='ADCP')
        z = uo_ann_clim['z_l']
        u = uo_ann_clim.isel(xq=iplt_m)*100.
        ax[2].plot(u,z,label='MOM6')
        ax[2].axvline(0.0,color='grey',linewidth=0.75)
        ax[2].set_xlabel('$cm s^{-1}$')
        ax[2].set_title('Annual Mean')

        ax[2].legend()
        pfile = 'u_ann.' + pos + '.png'
        plt.savefig(os.path.join(path_plt_out,pfile))

    plt.close('all')

    print('{} was run successfully!'.format(os.path.basename(__file__)))

    return

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()
