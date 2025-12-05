#!/usr/bin/env python

import io
import glob
import xarray as xr
import numpy
import nc_time_axis
import matplotlib.pyplot as plt
import os, yaml
from mom6_tools.m6toolbox import cime_xmlquery
from ncar_jobqueue import NCARCluster
from dask.distributed import Client
from datetime import datetime

try: import argparse
except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')

def options():
  parser = argparse.ArgumentParser(description='''Script for plotting time-series of transports across vertical sections.''')
  parser.add_argument('diag_config_yml_path', type=str, help='''Full path to the yaml file  \
    describing the run and diagnostics to be performed.''')

  parser.add_argument('-l','--label',    type=str, default='', help='''Label to add to the plot.''')
  parser.add_argument('-o','--outdir',   type=str, default='PNG/Transports', help='''Directory in which to place plots.''')
  parser.add_argument('-sd','--start_date',  type=str, default='0001-01-01',  help='''Start year to plot (default=0001-01-01)''')
  parser.add_argument('-ed','--end_date',   type=str, default='0100-12-31', help='''Final year to plot (default=0100-12-31)''')
  parser.add_argument('-nw','--number_of_workers',  type=int, default=1, help='''Number of workers to use (default=1).''')
  parser.add_argument('-save_ncfile', help='''Save a netCDF file with transport data''', action="store_true")
  parser.add_argument('-debug', help='''Add priting statements for debugging purposes''', action="store_true")
  args = parser.parse_args()
  return args

class Transport():
  def __init__(self, args, sections_dict, section, ylim=None, zlim=None, mks2Sv=True, debug=False):
    var = sections_dict[section][0]
    obs = sections_dict[section][1]
    label = section
    debug = debug or args.debug
    if debug: print('\n')
    if debug: print('##################################')
    if debug: print('Processing ', section)
    if debug: print('var ', var)
    if debug: print('obs ', obs)
    if debug: print('ylim ', ylim)
    if debug: print('##################################')

    # List all section* files in args.infile
    #full_path = args.infile+args.casename+'.mom6.'+section+'*'
    full_path = args.infile+args.casename+'.mom6.'+section+'*'
    if debug: print('Full path ', full_path)
    files = [f for f in glob.glob(full_path)]
    #files = [f for f in glob.glob(full_path, recursive=True)]
    tiles = [] # list with 'tiles' numbers. These change depending on the total # of PE
    for f in files:
      tiles.append(f[-4::])

    # Removing duplicates
    tiles = list(set(tiles))
    if debug: print('Tiles ', tiles)

    self.section = section
    self.var = var
    if label != None: self.label = label
    else: self.label = section
    self.ylim = ylim
    if debug: print('Start date {}; End date {}'.format(args.start_date, args.end_date))
    missing_var = True
    # loop over tiles
    for t in range(len(tiles)):
      inFileName = '{}.mom6.{}*nc.{}'.format(args.infile+args.casename, section, str(tiles[t]))
      if debug: print('inFileName {}, variable {}'.format(inFileName,var))
      ds = xr.open_mfdataset(inFileName, combine='by_coords', parallel=args.parallel)
      #ds['time'] = ds.indexes['time'].to_datetimeindex()
      if debug: print(ds)
      if var in ds.variables:
        missing_var = False
        if t == 0: total = numpy.ones(ds.variables[var][:].shape[0])*0.0
        if zlim is None: trans = ds.variables[var].sum(axis=1)  # Depth summation
        else:
          vardims = ds.variables[var].dims
          zdimname = vardims[1]
          z_l = ds.variables[zdimname].values[:]
          trans = ds.variables[var][:,(z_l>zlim[0]) & (z_l<zlim[1])].sum(axis=1)  # Limited depth summation

        if   var == 'umo': total = total + trans.sum(axis=1).squeeze().values
        elif var == 'vmo': total = total + trans.sum(axis=2).squeeze().values
        else: raise ValueError('Unknown variable name')
      else:
        if t == 0: total = 0.0
        total = total + 0.0

      # load time
      if 'time' in ds.variables:
        time = ds.variables['time'].values  # in years
        if args.debug: print('time[0],time[-1]',time[0],time[-1])
      else:
        raise ValueError('Variable time is missing')

    if mks2Sv == True: total = total * 1.e-9

    if missing_var:
      raise ValueError('Variable does not exist. Please verify that you assigned the right variable for this section.')

    self.data = total
    self.time = time
    if args.casename != '':  self.casename = args.casename + ' ' + args.label
    else: self.casename = ds.title + ' ' + args.label
    return

def plotPanel(section,n,observedFlows=None,colorCode=True):
    ax = plt.subplot(7,3,n+1)
    color = '#c3c3c3'; obsLabel = None
    if section.label in observedFlows.keys():
      #if isinstance(observedFlows[section.label],tuple):
      if isinstance(observedFlows[section.label][1:],list) and isinstance(observedFlows[section.label][1:][0],float):
        if colorCode == True:
          if min(observedFlows[section.label][1:]) <= section.data.mean() <= max(observedFlows[section.label][1:]):
            color = '#90ee90'
          else: color = '#f26161'
        obsLabel = str(min(observedFlows[section.label][1:])) + ' to ' + str(max(observedFlows[section.label][1:]))
      else: obsLabel = str(observedFlows[section.label][1:])
    plt.plot(section.time,section.data,color=color, lw=2)
    plt.title(section.label,fontsize=12)
    plt.text(0.04,0.11,'Mean = '+'{0:.2f}'.format(section.data.mean()),transform=ax.transAxes,fontsize=10)
    if obsLabel is not None: plt.text(0.04,0.04,'Obs. = '+obsLabel,transform=ax.transAxes,fontsize=10)
    if section.ylim is not None: plt.ylim(section.ylim)
    if n in [0,3,6,9,12,15,18]: plt.ylabel('Transport (Sv)')

def main(stream=False):

  start = datetime.now()
  # Get options
  args = options()
  if args.save_ncfile:
    if not os.path.isdir('ncfiles'):
      print('Creating a directory to place netCDF file (ncfiles)... \n')
      os.system('mkdir ncfiles')

  nw = args.number_of_workers
  # Read in the yaml file
  diag_config_yml = yaml.load(open(args.diag_config_yml_path,'r'), Loader=yaml.Loader)
  # load sections where transports are computed online
  sections = diag_config_yml['Transports']['sections']
  caseroot = diag_config_yml['Case']['CASEROOT']
  args.casename = cime_xmlquery(caseroot, 'CASE')
  DOUT_S = cime_xmlquery(caseroot, 'DOUT_S')
  if DOUT_S:
    OUTDIR = cime_xmlquery(caseroot, 'DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = cime_xmlquery(caseroot, 'RUNDIR')

  parallel = False
  if nw > 1:
    parallel = True
    cluster = NCARCluster()
    cluster.scale(nw)
    client = Client(cluster)

  args.parallel = parallel
  args.infile = OUTDIR
  if args.infile[-1] != '/': args.infile = args.infile+'/'
  print('Output directory is:', args.infile)
  print('Casename is:', args.casename)
  print('Number of workers to be used (nw > 1 means parallel=True):', nw)

  reference = 'Griffies et al., 2016: OMIP contribution to CMIP6: experimental and diagnostic protocol for the physical component of the Ocean Model Intercomparison Project. Geosci. Model. Dev., 9, 3231-3296. doi:10.5194/gmd-9-3231-2016'
  title = 'Griffies et al., 2016, Geosci. Model. Dev., 9, 3231-3296. doi:10.5194/gmd-9-3231-2016'
  observedFlows = {'reference':reference, 'title':title, 'sections':sections}

  plotSections = []

  # leaving this here to catch if start/end years outside the range of the dataset
  #res = Transport(args, sections, 'Agulhas_Section')

  for key in sections:
    print('Processing section {} ...\n'.format(key))
    startTime = datetime.now()
    try: res = Transport(args, sections, key); plotSections.append(res)
    except: print('\n WARNING: unable to process {}'.format(key))
    print('Time elasped: ', datetime.now() - startTime)

  if args.save_ncfile:
    print('Saving netCDF file with transports...\n')
    # create a dataaray
    labels = [];
    for n in range(len(plotSections)): labels.append(plotSections[n].label)
    var = numpy.zeros((len(plotSections),len(plotSections[0].time)))
    ds = xr.Dataset(data_vars={ 'transport' : (('sections', 'time'), var)},
                           coords={'sections': labels,
                                   'time': plotSections[0].time})
    for n in range(len(plotSections)):
      ds.transport.values[n,:] = plotSections[n].data

    ds.to_netcdf('ncfiles/'+args.casename+'_section_transports.nc')

  print('Plotting {} sections...\n'.format(len(plotSections)))
  imgbufs = []

  fig = plt.figure(figsize=(13,17))
  for n in range(0,len(plotSections)): plotPanel(plotSections[n],n,observedFlows=sections)
  fig.text(0.5,0.955,str(plotSections[n-1].casename),horizontalalignment='center',fontsize=14)
  fig.text(0.5,0.925,'Observations summarized in '+observedFlows['title'],horizontalalignment='center',fontsize=11)
  plt.subplots_adjust(hspace=0.3)
  if stream != None: fig.text(0.5,0.05,str('Generated by mom6-tools'),horizontalalignment='center',fontsize=12)
  #plt.show(block=False)

  if stream is True: objOut = io.BytesIO()
  else:
    if not os.path.isdir(args.outdir):
      os.system('mkdir -p '+args.outdir)
    objOut = args.outdir+'/'+args.casename+'_section_transports.png'
  plt.savefig(objOut)

  print('Total time elasped: ', datetime.now() - start)
  print('{} was run successfully!'.format(os.path.basename(__file__)))

  if stream is True: imgbufs.append(objOut)

  if stream is True:
    return imgbufs

if __name__ == '__main__':
  main()
