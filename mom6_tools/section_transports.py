#!/usr/bin/env python

import io
import glob
import xarray as xr
import numpy
import matplotlib.pyplot as plt
import os, yaml
from mom6_tools.DiagsCase import DiagsCase

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
  parser.add_argument('-save_ncfile', help='''Save a netCDF file with transport data''', action="store_true")
  parser.add_argument('-debug', help='''Add priting statements for debugging purposes''', action="store_true")
  cmdLineArgs = parser.parse_args()
  return cmdLineArgs

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
    #full_path = args.infile+args.case_name+'.mom6.'+section+'*'
    full_path = args.infile+args.case_name+'.mom6.'+section+'.*.nc.????'
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
      inFileName = '{}.mom6.{}.*.nc.{}'.format(args.infile+args.case_name, section, str(tiles[t]))
      if debug: print('inFileName {}, variable {}'.format(inFileName,var))
      rootGroup = xr.open_mfdataset(inFileName, combine='by_coords')
      #rootGroup['time'] = rootGroup.indexes['time'].to_datetimeindex()
      if debug: print(rootGroup)
      # select time range requested
      rootGroup = rootGroup.sel(time=slice(args.start_date, args.end_date))
      if debug: print('sd,ed,time[0],time[-1]',args.start_date,args.end_date, rootGroup.time.data[0], rootGroup.time.data[-1])
      # yearly mean
      rootGroup = rootGroup.resample(time="1Y", closed='left', keep_attrs=True).mean(dim='time', keep_attrs=True).load()
      if debug: print('Yearly mean: sd,ed,time[0],time[-1]',args.start_date,args.end_date, rootGroup.time.data[0], rootGroup.time.data[-1])
      if var in rootGroup.variables:
        missing_var = False
        if t == 0: total = numpy.ones(rootGroup.variables[var][:].shape[0])*0.0
        if zlim is None: trans = rootGroup.variables[var].sum(axis=1)  # Depth summation
        else:
          vardims = rootGroup.variables[var].dims
          zdimname = vardims[1]
          z_l = rootGroup.variables[zdimname].values[:]
          trans = rootGroup.variables[var][:,(z_l>zlim[0]) & (z_l<zlim[1])].sum(axis=1)  # Limited depth summation

        if   var == 'umo': total = total + trans.sum(axis=1).squeeze().values
        elif var == 'vmo': total = total + trans.sum(axis=2).squeeze().values
        else: raise ValueError('Unknown variable name')
      else:
        if t == 0: total = 0.0
        total = total + 0.0

      # load time
      if 'time' in rootGroup.variables:
        time = rootGroup.variables['time'].values  # in years
        if args.debug: print('time[0],time[-1]',time[0],time[-1])
      else:
        raise ValueError('Variable time is missing')

    if mks2Sv == True: total = total * 1.e-9

    if missing_var:
      raise ValueError('Variable does not exist. Please verify that you assigned the right variable for this section.')

    self.data = total
    self.time = time
    if args.case_name != '':  self.case_name = args.case_name + ' ' + args.label
    else: self.case_name = rootGroup.title + ' ' + args.label
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

  # Get options
  cmdLineArgs = options()
  if cmdLineArgs.save_ncfile:
    if not os.path.isdir('ncfiles'):
      print('Creating a directory to place figures (ncfiles)... \n')
      os.system('mkdir ncfiles')

  # Read in the yaml file
  diag_config_yml = yaml.load(open(cmdLineArgs.diag_config_yml_path,'r'), Loader=yaml.Loader)
  # load sections where transports are computed online
  sections = diag_config_yml['Transports']['sections']
  # Create the case instance
  dcase = DiagsCase(diag_config_yml['Case'])
  cmdLineArgs.case_name = dcase.casename
  DOUT_S = dcase.get_value('DOUT_S')
  if DOUT_S:
    OUTDIR = dcase.get_value('DOUT_S_ROOT')+'/ocn/hist/'
  else:
    OUTDIR = dcase.get_value('RUNDIR')

  cmdLineArgs.infile = OUTDIR
  if cmdLineArgs.infile[-1] != '/': cmdLineArgs.infile = cmdLineArgs.infile+'/'
  if cmdLineArgs.debug:
    print('Output directory is:', cmdLineArgs.infile)
    print('Casename is:', cmdLineArgs.case_name)


  reference = 'Griffies et al., 2016: OMIP contribution to CMIP6: experimental and diagnostic protocol for the physical component of the Ocean Model Intercomparison Project. Geosci. Model. Dev., 9, 3231-3296. doi:10.5194/gmd-9-3231-2016'
  title = 'Griffies et al., 2016, Geosci. Model. Dev., 9, 3231-3296. doi:10.5194/gmd-9-3231-2016'
  observedFlows = {'reference':reference, 'title':title, 'sections':sections}

  plotSections = []

  # leaving this here to catch if start/end years outside the range of the dataset
  res = Transport(cmdLineArgs, sections, 'Agulhas_Section')

  for key in sections:
    try: res = Transport(cmdLineArgs, sections, key); plotSections.append(res)
    except: print('\n WARNING: unable to process {}'.format(key))

  if cmdLineArgs.save_ncfile:
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

    ds.to_netcdf('ncfiles/'+cmdLineArgs.case_name+'_section_transports.nc')

  print('Plotting {} sections...\n'.format(len(plotSections)))
  imgbufs = []

  fig = plt.figure(figsize=(13,17))
  for n in range(0,len(plotSections)): plotPanel(plotSections[n],n,observedFlows=sections)
  fig.text(0.5,0.955,str(plotSections[n-1].case_name),horizontalalignment='center',fontsize=14)
  fig.text(0.5,0.925,'Observations summarized in '+observedFlows['title'],horizontalalignment='center',fontsize=11)
  plt.subplots_adjust(hspace=0.3)
  if stream != None: fig.text(0.5,0.05,str('Generated by mom6-tools'),horizontalalignment='center',fontsize=12)
  #plt.show(block=False)

  if stream is True: objOut = io.BytesIO()
  else:
    if not os.path.isdir(cmdLineArgs.outdir):
      os.system('mkdir -p '+cmdLineArgs.outdir)
    objOut = cmdLineArgs.outdir+'/'+cmdLineArgs.case_name+'_section_transports.png'
  plt.savefig(objOut)

  print('{} was run successfully!'.format(os.path.basename(__file__)))

  if stream is True: imgbufs.append(objOut)

  if stream is True:
    return imgbufs

if __name__ == '__main__':
  main()
