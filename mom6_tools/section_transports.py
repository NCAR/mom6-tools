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
  def __init__(self, args, section, var, label=None, ylim=None, zlim=None, mks2Sv=True, debug=False):
    debug = debug or args.debug
    if debug: print('Processing ', section)

    # List all section* files in args.infile
    full_path = args.infile+args.case_name+'.mom6.'+section+'*'
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
      inFileName = '{}.mom6.{}_*.nc.{}'.format(args.infile+args.case_name, section, str(tiles[t]))
      if debug: print('inFileName {}, variable {}'.format(inFileName,var))
      rootGroup = xr.open_mfdataset(inFileName, combine='by_coords')
      #rootGroup['time'] = rootGroup.indexes['time'].to_datetimeindex()
      if debug: print(rootGroup)
      # select time range requested
      rootGroup = rootGroup.sel(time=slice(args.start_date, args.end_date))
      if debug: print('sd,ed,time',args.start_date,args.end_date, rootGroup.time.data)
      # yearly mean
      rootGroup = rootGroup.resample(time="1Y", closed='left', keep_attrs=True).mean(dim='time', keep_attrs=True).load()
      if debug: print('Yearly mean: sd,ed,time',args.start_date,args.end_date, rootGroup.time.data)
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
        if args.debug: print('time',time)
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
    ax = plt.subplot(6,3,n+1)
    color = '#c3c3c3'; obsLabel = None
    if section.label in observedFlows.keys():
      if isinstance(observedFlows[section.label],tuple):
        if colorCode == True:
          if min(observedFlows[section.label]) <= section.data.mean() <= max(observedFlows[section.label]):
            color = '#90ee90'
          else: color = '#f26161'
        obsLabel = str(min(observedFlows[section.label])) + ' to ' + str(max(observedFlows[section.label]))
      else: obsLabel = str(observedFlows[section.label])
    plt.plot(section.time,section.data,color=color, lw=2)
    plt.title(section.label,fontsize=12)
    plt.text(0.04,0.11,'Mean = '+'{0:.2f}'.format(section.data.mean()),transform=ax.transAxes,fontsize=10)
    if obsLabel is not None: plt.text(0.04,0.04,'Obs. = '+obsLabel,transform=ax.transAxes,fontsize=10)
    if section.ylim is not None: plt.ylim(section.ylim)
    if n in [1,4,7,10,13,16]: plt.ylabel('Transport (Sv)')

def main(stream=False):

  # Get options
  cmdLineArgs = options()
  if cmdLineArgs.save_ncfile:
    if not os.path.isdir('ncfiles'):
      print('Creating a directory to place figures (ncfiles)... \n')
      os.system('mkdir ncfiles')

  # Read in the yaml file
  diag_config_yml = yaml.load(open(cmdLineArgs.diag_config_yml_path,'r'), Loader=yaml.Loader)
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
  observedFlows = {'reference':reference, 'title':title,
                   'Agulhas':(164.4,182.2), 'Barents Opening':2.0, 'Bering Strait':(0.7,1.1), 'Davis Strait':(-2.1,-1.1), 'Denmark Strait':(-4.8,-2.0),
                   'Drake Passage':(164.4,182.2), 'English Channel':(0.01,0.1), 'Faroe-Scotland':(0.8,1.0), 'Florida-Bahamas':(28.9,34.3),
                   'Fram Strait':(-4.7,0.7), 'Gibraltar Strait':0.11, 'Gibraltar Strait':(-1.0, 1.0), 'Iceland-Faroe':(4.35,4.85),
                   'Indonesian Throughflow':(-15.,-13.), 'Mozambique Channel':(-25.6,-7.8), 'Pacific Equatorial Undercurrent':(24.5,28.3),
                   'Taiwan-Luzon Strait':(-3.0,-1.8), 'Windward Passage':(-15.,5.)}
  # GMM, we need some estimated transport for the following:
  # Bab_al_mandeb_Strait
  # Iceland-Norway
  # Hormuz Strait

  plotSections = []

  # leaving this here to catch if start/end years outside the range of the dataset
  res = Transport(cmdLineArgs,'agulhas_section','umo',label='Agulhas',ylim=(100,200))

  try: res = Transport(cmdLineArgs,'agulhas_section','umo',label='Agulhas',ylim=(100,250)); plotSections.append(res)
  except: print('WARNING: unable to process Agulhas_section')

  try: res = Transport(cmdLineArgs,'Bab_al_mandeb_Strait','umo',label='Bab al mandeb Strait',ylim=(-0.5, 0.5)); plotSections.append(res)
  except: print('WARNING: unable to process Bab Al Mandeb Strait')

  try: res = Transport(cmdLineArgs,'Bering_Strait','vmo',label='Bering Strait',ylim=(-2,3)); plotSections.append(res)
  except: print('WARNING: unable to process Bering_Strait')

  try: res = Transport(cmdLineArgs,'Barents_opening','vmo',label='Barents Opening',ylim=(-1,9)); plotSections.append(res)
  except: print('WARNING: unable to process Barents_opening')

  try: res = Transport(cmdLineArgs,'Davis_Strait','vmo',label='Davis Strait',ylim=(-5.0,0.5)); plotSections.append(res)
  except: print('WARNING: unable to process Davis_Strait')

  try: res = Transport(cmdLineArgs,'Denmark_Strait','vmo',label='Denmark Strait',ylim=(-12,2)); plotSections.append(res)
  except: print('WARNING: unable to process Denmark_Strait')

  try: res = Transport(cmdLineArgs,'Drake_Passage','umo',label='Drake Passage',ylim=(100,250)); plotSections.append(res)
  except: print('WARNING: unable to process Drake_Passage')

  try: res = Transport(cmdLineArgs,'English_Channel','umo',label='English Channel',ylim=(-0.4,0.4)); plotSections.append(res)
  except: print('WARNING: unable to process English_Channel')

  #try: res = Transport(cmdLineArgs,'Faroe_Scotland','umo',label='Faroe-Scotland',ylim=(-5,12)); plotSections.append(res)
  #except: print('WARNING: unable to process Faroe_Scotland')

  try: res = Transport(cmdLineArgs,'Florida_Bahamas','vmo',label='Florida-Bahamas',ylim=(5,35)); plotSections.append(res)
  except: print('WARNING: unable to process Florida_Bahamas')

  try: res = Transport(cmdLineArgs,'Fram_Strait','vmo',label='Fram Strait',ylim=(-8,4)); plotSections.append(res)
  except: print('WARNING: unable to process Fram_Strait')

  try: res = Transport(cmdLineArgs,'Gibraltar_Strait','umo',label='Gibraltar Strait',ylim=(-1.0,1.0)); plotSections.append(res)
  except: print('WARNING: unable to process Gibraltar_Strait')

  try: res = Transport(cmdLineArgs,'Hormuz_Strait','umo',label='Hormuz Strait',ylim=(-0.5,0.5)); plotSections.append(res)
  except: print('WARNING: unable to process Hormuz_Strait')

  #try: res = Transport(cmdLineArgs,['Iceland_Faroe_U','Iceland_Faroe_V'],['umo','vmo'],label='Iceland-Faroe'); plotSections.append(res)
  #except: print('WARNING: unable to process Iceland_Faroe_U and Iceland_Faroe_V')

  try: res = Transport(cmdLineArgs,'Iceland_Norway','vmo',label='Iceland-Norway',ylim=(-5,15)); plotSections.append(res)
  except: print('WARNING: unable to process Iceland_Norway')

  try: res = Transport(cmdLineArgs,'Indonesian_Throughflow','vmo',label='Indonesian Throughflow',ylim=(-40,10)); plotSections.append(res)
  except: print('WARNING: unable to process Indonesian_Throughflow')

  try: res = Transport(cmdLineArgs,'Mozambique_Channel','vmo',label='Mozambique Channel',ylim=(-50,10)); plotSections.append(res)
  except: print('WARNING: unable to process Mozambique_Channel')

  try: res = Transport(cmdLineArgs,'Pacific_undercurrent','umo',label='Pacific Equatorial Undercurrent',ylim=None, zlim=(0,350)); plotSections.append(res)
  except: print('WARNING: unable to process Pacific_undercurrent')

  try: res = Transport(cmdLineArgs,'Taiwan_Luzon','umo',label='Taiwan-Luzon Strait',ylim=(-15,10)); plotSections.append(res)
  except: print('WARNING: unable to process Taiwan_Luzon')

  try: res = Transport(cmdLineArgs,'Windward_Passage','vmo',label='Windward Passage',ylim=(-10,10)); plotSections.append(res)
  except: print('WARNING: unable to process Windward_Passage')

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
  for n in range(0,len(plotSections)): plotPanel(plotSections[n],n,observedFlows=observedFlows)
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
