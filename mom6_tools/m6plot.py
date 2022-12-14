"""
A method for producing a standardized pseudo-color plot of 2D data
"""

import os
try:
  if os.environ['DISPLAY'] != None: pass
except:
  import matplotlib
  matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib
import matplotlib.path as mpath
import math
import numpy, numpy.matlib
import cartopy.crs as ccrs
import cartopy.feature
from mom6_tools import VerticalSplitScale
from mom6_tools import m6toolbox

try: from mpl_toolkits.basemap import Basemap
except: print('Basemap module not found. Some regional plots may not function properly')

try:
  import cmocean
except:
  if "HTTP_USER_AGENT" in os.environ.keys():
    pass
  else:
    print('cmocean module not found. Some color maps may not render properly')

from sys import modules


def plot_stats_da(stats, var, units, save=None):
  """
  Plot statistics computed by myStats_da for multiple basins.

  Parameters
  ----------
  stats : xarray.DataArray
    DataArray with statistics computed using myStats_da.

  title : str
   Title to be used in the plot.

  var : str
   Variable to used to compute stats.

  units : str
   Units of variable var.

  save : str, optional
    Name of file to save figure in. Default None.

  """
  import seaborn
  import matplotlib
  matplotlib.rcParams.update({'font.size': 15})
  seaborn.set()
  labels = ['Min', 'Max', 'Mean', 'Std', 'RMS']
  plt.figure(figsize=(12,14))
  for i in range(len(labels)):
    ax1 = plt.subplot(5,1,i+1)
    for r in range(len(stats.basin)):
      stats[r,i,:].plot(ax=ax1, marker='.', linewidth = 3, label=str(stats.basin[r].values));

    ax1.set_ylabel(str(labels[i]))
    if i == 0:
      plt.legend()
      plt.title(str(var) + ' ['+str(units)+']')
    else:
      plt.title('')
    if i < (len(labels) -1):
      ax1.set_xlabel('')

  if save is not None:
    plt.savefig(save)
    plt.close()

  return

def plot_stats_da_reg(stats, title, var, units):
  """
  Plot statistics computed by myStats_da for a specific region.

  Parameters
  ----------
  stats : xarray.DataArray
    DataArray with statistics computed using myStats_da.

  title : str
   Title to be used in the plot.

  var : str
   Variable to used to compute stats.

  units : str
   Units of variable var.
  """
  import seaborn
  matplotlib.rcParams.update({'font.size': 15})
  seaborn.set()
  plt.figure(figsize=(12,14))
  ax1 = plt.subplot(511)
  stats[0,:].plot(ax=ax1, marker='.'); ax1.set_ylabel('Min')
  plt.title(str(var) + ' ['+str(units)+'], Region = '+ str(title))
  ax1.set_xlabel('')
  ax2 = plt.subplot(512, sharex=ax1)
  stats[1,:].plot(ax=ax2, marker='.'); ax2.set_ylabel('Max')
  plt.title('')
  ax2.set_xlabel('')
  ax3 = plt.subplot(513, sharex=ax1)
  stats[2,:].plot(ax=ax3, marker='.'); ax3.set_ylabel('Mean')
  plt.title('')
  ax3.set_xlabel('')
  ax4 = plt.subplot(514, sharex=ax1)
  stats[3,:].plot(ax=ax4, marker='.'); ax4.set_ylabel('Std')
  plt.title('')
  ax4.set_xlabel('')
  ax5 = plt.subplot(515, sharex=ax1)
  stats[4,:].plot(ax=ax5, marker='.'); ax5.set_ylabel('RMS')
  plt.title('')
  if save is not None:
    plt.savefig(save)
    plt.close()

  return

def polarcomparison(field1, field2, grd, proj='SP', circle=True,
              clim = None, colormap = None, nbins = None, save=None,
              title1='', title2='', title3='A - B', suptitle='',
              ignore = None, debug=False, show=False, extend = None,
              sigma = 2.0, logscale = False, landcolor=[.5,.5,.5], interactive=False,
              dlim=None, dcolormap=None, dextend=None, addplabel=True):

  """Renders n-panel plot of two scalar field1(x,y) and field2(x,y) using polar projections.

  Parameters
  ----------
  field1 : 2D numpy array
    Scalar 2D array to be plotted and compared to field2.

  field2 : 2D numpy array
    Scalar 2D array to be plotted and compared to field1.

  grd : object with MOM6 grid data
    This is the output of module MOM6grid.

  title1 : str, optional
     The title to place at the top of panel 1. Default ''.

  title2 : str, optional
     The title to place at the top of panel 2. Default ''.

  title3 : str, optional
     The title to place at the top of panel 3. Default 'A-B'.

  addplabel : boolean, optional
     Adds a 'A:' or 'B:' to the title1 and title2. Default True.

  suptitle : str, optional
     The super-title to place at the top of the figure. Default ''.

  proj  : str, optional
    Type of projection: 'SP' = SouthPolarStereo (default) or 'NP' = NorthPolarStereo

  circle : boolean, optional
    If true, compute a circle in axes coordinates, which we can use as a boundary
    for the map

  clim  : tuple of (min,max), optional
     color range OR a list of contour levels. Default None

  colormap : str, optional
    The name of the colormap to use. Default None (choose using chooseColorMap)

  nbins : integer, optional
    The number of colors levels (used if clim is missing or only specifies the color range).

  ignore : float, optional
    A value to use as no-data (NaN). Default None

  save : str, optional
    Name of file to save figure in. Default None (do not save figure)

  debug : boolean, optional
    If true, report stuff for debugging. Default False

  show : boolean, optional
    If true, causes the figure to appear on screen. Used for testing. Default False.

  extend : str, optional
    Can be one of 'both', 'neither', 'max', 'min'. Default None

  logscale : boolean, optional
    If true, use logaritmic coloring scheme. Default False

  sigma : float, optional
    Range for difference plot autocolor levels. Default is to span a 2. sigma range

  title  : str, optional
    The title to place at the top of the panel. Default ''

  landcolor : RGB tuple, optional
    An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].

  dlim  : tuple (min,max)
    A tuple of (min,max) color range OR a list of contour levels for the difference plot. Default None.

  dcolormap  : str, optional
    The name of the colormap to use for the difference plot. Default None.

  dextend  : str, optional
     For the difference colorbar. Can be one of 'both', 'neither', 'max', 'min'. Default None.

  interactive : boolean, optional
    If true, adds interactive features such as zoom, close and cursor. Default False.

  Returns
  -------
  """

  if (field1.shape)!=(field2.shape): raise Exception('field1 and field2 must be the same shape')


  xCoord, yCoord = createXYcoords(field1, grd.geolon, grd.geolat)

  if proj == 'SP':
    proj = ccrs.SouthPolarStereo()
    extent = [-180, 180, -90, -50]
  else: # NP
    proj = ccrs.NorthPolarStereo()
    extent = [-180, 180, 50, 90]

  # Establish ranges for sectors
  #lonRange=(grd.geolon.min(), grd.geolon.max()); latRange=(extent[2], extent[3])
  lonRange=(-360, 360.); latRange=(extent[2], extent[3])

  # Diagnose statistics
  if ignore is not None:
    maskedField1 = numpy.ma.masked_array(field1, mask=[field1==ignore])
    maskedField2 = numpy.ma.masked_array(field2, mask=[field2==ignore])
  else:
    maskedField1 = regionalMasking(field1, yCoord, xCoord, latRange, lonRange)
    maskedField2 = regionalMasking(field2, yCoord, xCoord, latRange, lonRange)

  try:
    area = grd.area_t
  except:
    area = grd.areacello

  s1Min, s1Max, s1Mean, s1Std, s1RMS = myStats(maskedField1, area, debug=debug)
  s2Min, s2Max, s2Mean, s2Std, s2RMS = myStats(maskedField2, area, debug=debug)
  dMin, dMax, dMean, dStd, dRMS = myStats(maskedField1 - maskedField2, area, debug=debug)

  if s1Mean is not None: dRxy = corr(maskedField1 - s1Mean, maskedField2 - s2Mean, area)
  else: dRxy = None

  s12Min = min(s1Min, s2Min); s12Max = max(s1Max, s2Max)
  xLims = boundaryStats(xCoord); yLims = boundaryStats(yCoord)
  if debug:
    print('s1: min, max, mean =', s1Min, s1Max, s1Mean)
    print('s2: min, max, mean =', s2Min, s2Max, s2Mean)
    print('s12: min, max =', s12Min, s12Max)

  # Choose colormap
  if nbins is None and (clim is None or len(clim)==2): cBins=35
  else: cBins=nbins
  if nbins is None and (dlim is None or len(dlim)==2): nbins=35
  if colormap is None: colormap = chooseColorMap(s12Min, s12Max)
  cmap, norm, extend = chooseColorLevels(s12Min, s12Max, colormap, clim=clim, nbins=cBins, extend=extend)

  if addplabel: preTitleA = 'A: '; preTitleB = 'B: '
  else: preTitleA = ''; preTitleB = ''

  # hard-coded number of panels, aspect and resolution
  npanels=3
  #aspect=None
  #resolution=None
  #axis = None

  #  fig=setFigureSize(aspect, resolution, npanels=npanels, debug=debug)
  fig = plt.figure(figsize=[8, 24])
  # panel 1
  ax = plt.subplot(npanels,1,1, projection=proj)
  ax.set_extent(extent, ccrs.PlateCarree())
  ax.add_feature(cartopy.feature.LAND)
  ax.gridlines()

  if circle:
    circle_value = get_circle()
    ax.set_boundary(circle_value, transform=ax.transAxes)

  cs = ax.pcolormesh(xCoord,yCoord,maskedField1,transform=ccrs.PlateCarree(),cmap=cmap,
                     shading='flat', norm=norm)

  def add_features(fig, ax, cs, extend, landcolor):
    """ Adds some features to the map """
    fig.colorbar(cs,fraction=.08, pad=0.02, extend=extend)
    # Add Land
    ax.add_feature( cartopy.feature.LAND, zorder=1, edgecolor='none', facecolor=landcolor)
    # add Ocean
    ax.add_feature(cartopy.feature.OCEAN)
    # Add coastline
    ax.coastlines(color='black')
    # Add lat lon rings
    ax.gridlines(alpha=0.1,color='black')
    return

  add_features(fig, ax, cs, extend, landcolor)

  if interactive: addStatusBar(xCoord, yCoord, maskedField1)
  ax.set_xticklabels([''])
  annotateStats(ax, s1Min, s1Max, s1Mean, s1Std, s1RMS, webversion=False)
  if len(title1)>0: ax.set_title(preTitleA+title1)

  # panel 2
  ax = plt.subplot(npanels,1,2, projection=proj)
  ax.set_extent(extent, ccrs.PlateCarree())
  ax.add_feature(cartopy.feature.LAND)
  ax.gridlines()

  if circle:
    circle_value = get_circle()
    ax.set_boundary(circle_value, transform=ax.transAxes)

  cs = ax.pcolormesh(xCoord,yCoord,maskedField2,transform=ccrs.PlateCarree(),cmap=cmap,
                     shading='flat', norm=norm)

  add_features(fig, ax, cs, extend, landcolor)

  if interactive: addStatusBar(xCoord, yCoord, maskedField2)
  if npanels>2: ax.set_xticklabels([''])
  annotateStats(ax, s2Min, s2Max, s2Mean, s2Std, s2RMS, webversion=False)
  if len(title2)>0: ax.set_title(preTitleB+title2)

  # panel 3

  ax = plt.subplot(npanels,1,npanels, projection=proj)
  ax.set_extent(extent, ccrs.PlateCarree())
  ax.add_feature(cartopy.feature.LAND)
  ax.gridlines()

  if circle:
    circle_value = get_circle()
    ax.set_boundary(circle_value, transform=ax.transAxes)

  if dcolormap is None: dcolormap = chooseColorMap(dMin, dMax)
  if dlim is None and dStd>0:
    cmap, norm, dextend = chooseColorLevels(dMean-sigma*dStd, dMean+sigma*dStd, dcolormap, clim=dlim, nbins=nbins, \
                                            extend=dextend, autocenter=True)
  else:
    cmap, norm, dextend = chooseColorLevels(dMin, dMax, dcolormap, clim=dlim, nbins=nbins, extend=dextend, autocenter=True)



  cs = ax.pcolormesh(xCoord,yCoord,maskedField1 - maskedField2,transform=ccrs.PlateCarree(),cmap=cmap, shading='flat', norm=norm)

  if interactive: addStatusBar(xCoord, yCoord, maskedField1 - maskedField2)
  if dextend is None: dextend = extend
  add_features(fig, ax, cs, dextend, landcolor)
  annotateStats(ax, dMin, dMax, dMean, dStd, dRMS, webversion=False)
  if len(title3)>0: ax.set_title(title3)

  plt.suptitle(suptitle,y=1.0)

  if save is not None: plt.savefig(save,bbox_inches='tight')
  if interactive: addInteractiveCallbacks()
  if show: plt.show(block=False)


  return

def get_circle():
  """
  Compute a circle in axes coordinates, which we can use as a boundary
  for the map. We can pan/zoom as much as we like - the boundary will be
  permanently circular.
  """

  theta = numpy.linspace(0, 2*numpy.pi, 100)
  center, radius = [0.5, 0.5], 0.5
  verts = numpy.vstack([numpy.sin(theta), numpy.cos(theta)]).T

  return mpath.Path(verts * radius + center)

def polarplot(field, grd, proj='SP', contour=None, circle=True,
              clim = None, colormap = None, nbins = None, save=None,
              ignore = None, debug=False, show=False, extend = None,
              sigma = 2.0, logscale = False, title = '', aspect=[16,9],
              landcolor=[.5,.5,.5], resolution=576, axis=None, grid=True,
              annotate=True, clabel = ''):

  """Renders plot of scalar field(x,y) using polar projections.

  Parameters
  ----------
  field : 2D numpy array
    Scalar 2D array to be plotted

  grd : object with MOM6 grid data
    This is the output of module MOM6grid.

  contour : float, optional
    If defined, coutours this value in the figure.

  proj  : str, optional
    Type of projection: 'SP' = SouthPolarStereo (default) or 'NP' = NorthPolarStereo

  circle : boolean, optional
    If true, compute a circle in axes coordinates, which we can use as a boundary
    for the map

  clim  : tuple of (min,max), optional
     color range OR a list of contour levels. Default None

  colormap : str, optional
    The name of the colormap to use. Default None (choose using chooseColorMap)

  nbins : integer, optional
    The number of colors levels (used if clim is missing or only specifies the color range).

  ignore : float, optional
    A value to use as no-data (NaN). Default None

  save : str, optional
    Name of file to save figure in. Default None (do not save figure)

  debug : boolean, optional
    If true, report stuff for debugging. Default False

  show : boolean, optional
    If true, causes the figure to appear on screen. Used for testing. Default False.

  extend : str, optional
    Can be one of 'both', 'neither', 'max', 'min'. Default None

  logscale : boolean, optional
    If true, use logaritmic coloring scheme. Default False

  sigma : float, optional
    Range for difference plot autocolor levels. Default is to span a 2. sigma range

  title  : str, optional
    The title to place at the top of the panel. Default ''

  clabel  : str, optional
    Colorbar label. Default ''

  landcolor : RGB tuple, optional
    An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].

  aspect  : tuple, optional
     The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].

  resolution  : integet, optional
     The vertical resolution of the figure given in pixels. Default 720.

  axis :  matplotlib.axes._subplots.AxesSubplot
     The axis handle to plot to. Default None.

  grid : boolean, optional
     If True, plot grid. Default is True.

  annotate : boolean, optional
     If True, write stats in the figure. Default is True.


  Returns
  -------
  """

  xCoord, yCoord = createXYcoords(field, grd.geolon, grd.geolat)

  if proj == 'SP':
    proj = ccrs.SouthPolarStereo()
    extent = [-180, 180, -90, -50]
  else: # NP
    proj = ccrs.NorthPolarStereo()
    extent = [-180, 180, 50, 90]

  # Establish ranges for sectors
  #lonRange=(grd.geolon.min(), grd.geolon.max()); latRange=(extent[2], extent[3]);
  lonRange=(-360, 360.); latRange=(extent[2], extent[3])

  # Diagnose statistics
  if ignore is not None: maskedField = numpy.ma.masked_array(field, mask=[field==ignore])
  else:
    maskedField = regionalMasking(field, yCoord, xCoord, latRange, lonRange)

  try:
    area = grd.area_t
  except:
    area = grd.areacello

  sMin, sMax, sMean, sStd, sRMS = myStats(maskedField, area, debug=debug)

  # Choose colormap
  if nbins is None and (clim is None or len(clim)==2): nbins=35
  if colormap is None: colormap = chooseColorMap(sMin, sMax)
  if clim is None and sStd is not None:
    cmap, norm, extend = chooseColorLevels(sMean-sigma*sStd, sMean+sigma*sStd, colormap, clim=clim, nbins=nbins, extend=extend, logscale=logscale)
  else:
    cmap, norm, extend = chooseColorLevels(sMin, sMax, colormap, clim=clim, nbins=nbins, extend=extend, logscale=logscale)

  if axis is None:
    setFigureSize(aspect, resolution, debug=debug)
    #plt.gcf().subplots_adjust(left=.08, right=.99, wspace=0, bottom=.09, top=.9, hspace=0)
    axis = plt.gca(projection=proj)

  axis.set_extent(extent, ccrs.PlateCarree())
  axis.add_feature(cartopy.feature.LAND)
  if grid:
    axis.gridlines()

  if circle:
    circle_value = get_circle()
    axis.set_boundary(circle_value, transform=axis.transAxes)

  cs = axis.pcolormesh(xCoord,yCoord,maskedField,transform=ccrs.PlateCarree(),cmap=cmap,
                       shading='flat', norm=norm)
  #plt.colorbar(cs, ax=axis, fraction=.08, pad=0.02, orientation='horizontal')
  cb = plt.colorbar(cs, ax=axis, orientation='horizontal', pad=0.02)
  cb.set_label(clabel, fontsize=14)
  # Add Land
  axis.add_feature( cartopy.feature.LAND, zorder=1, edgecolor='none', facecolor=landcolor) #fae5c9')
  # add Ocean
  #axis.add_feature(cartopy.feature.OCEAN)
  # Add coastline
  axis.coastlines(color='black')
  # Add lat lon rings
  #axis.gridlines(alpha='0.1',color='black')
  if contour is not None:
    axis.contour(grd.geolon,grd.geolat,maskedField,[contour],colors='red', transform=ccrs.PlateCarree())

  if annotate:
    axis.annotate('max=%.5g\nmin=%.5g'%(sMax,sMin), xy=(0.0,1.01), xycoords='axes fraction', verticalalignment='bottom', fontsize=12)
    axis.annotate('mean=%.5g\nrms=%.5g'%(sMean,sRMS), xy=(1.0,1.01), xycoords='axes fraction',verticalalignment='bottom', horizontalalignment='right', fontsize=12)
    axis.annotate(' sd=%.5g\n'%(sStd), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='left', fontsize=12)

  if len(title)>0:  axis.set_title(title, fontsize=12)
  if save is not None: plt.savefig(save)
  if show: plt.show(block=False)
  plt.tight_layout()

  return


def xyplot(field, x=None, y=None, area=None,
  xlabel=None, xunits=None, ylabel=None, yunits=None,
  title='', suptitle='',
  clim=None, colormap=None, extend=None, centerlabels=False,
  nbins=None, landcolor=[.5,.5,.5], axis=None, add_cbar=True,
  aspect=[16,9], resolution=576, sigma=2., annotate=True,
  ignore=None, save=None, debug=False, show=False, interactive=False, logscale=False):
  """
  Renders plot of scalar field, field(x,y).

  Arguments:
  field        Scalar 2D array to be plotted.
  x            x coordinate (1D or 2D array). If x is the same size as field then x is treated as
               the cell center coordinates.
  y            y coordinate (1D or 2D array). If x is the same size as field then y is treated as
               the cell center coordinates.
  area         2D array of cell areas (used for statistics). Default None.
  xlabel       The label for the x axis. Default 'Longitude'.
  xunits       The units for the x axis. Default 'degrees E'.
  ylabel       The label for the y axis. Default 'Latitude'.
  yunits       The units for the y axis. Default 'degrees N'.
  title        The title to place at the top of the panel. Default ''.
  suptitle     The super-title to place at the top of the figure. Default ''.
  clim         A tuple of (min,max) color range OR a list of contour levels. Default None.
  sigma         Sigma range for difference plot autocolori levels. Default is to span a 2. sigma range
  annotate     If True (default), annotates stats (mi, max, etc) in the plot.
  colormap     The name of the colormap to use. Default None.
  add_cbar     If true, the colorbar is added to the panel. Otherwise, the function will return the color
               scale object that be used to create a colorbar. Default is True.
  extend       Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels If True, will move the colorbar labels to the middle of the interval. Default False.
  nbins        The number of colors levels (used is clim is missing or only specifies the color range).
  landcolor    An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  aspect       The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution   The vertical resolution of the figure given in pixels. Default 720.
  axis         The axis handle to plot to. Default None.
  ignore       A value to use as no-data (NaN). Default None.
  save         Name of file to save figure in. Default None.
  debug        If true, report stuff for debugging. Default False.
  show         If true, causes the figure to appear on screen. Used for testing. Default False.
  interactive  If true, adds interactive features such as zoom, close and cursor. Default False.
  logscale     If true, use logaritmic coloring scheme. Default False.
  """

  # Create coordinates if not provided
  xlabel, xunits, ylabel, yunits = createXYlabels(x, y, xlabel, xunits, ylabel, yunits)
  if debug: print('x,y label/units=',xlabel,xunits,ylabel,yunits)
  xCoord, yCoord = createXYcoords(field, x, y)

  # Diagnose statistics
  if ignore is not None: maskedField = numpy.ma.masked_array(field, mask=[field==ignore])
  else: maskedField = field.copy()
  sMin, sMax, sMean, sStd, sRMS = myStats(maskedField, area, debug=debug)
  xLims = boundaryStats(xCoord)
  yLims = boundaryStats(yCoord)

  # Choose colormap
  if nbins is None and (clim is None or len(clim)==2): nbins=35
  if colormap is None: colormap = chooseColorMap(sMin, sMax)
  if clim is None and sStd is not None:
    cmap, norm, extend = chooseColorLevels(sMean-sigma*sStd, sMean+sigma*sStd, colormap, clim=clim, nbins=nbins, extend=extend, logscale=logscale)
  else:
    cmap, norm, extend = chooseColorLevels(sMin, sMax, colormap, clim=clim, nbins=nbins, extend=extend, logscale=logscale)

  if axis is None:
    setFigureSize(aspect, resolution, debug=debug)
    #plt.gcf().subplots_adjust(left=.08, right=.99, wspace=0, bottom=.09, top=.9, hspace=0)
    axis = plt.gca()
  cs = axis.pcolormesh(xCoord, yCoord, maskedField, cmap=cmap, norm=norm)
  if interactive: addStatusBar(xCoord, yCoord, maskedField)
  if add_cbar: cb = plt.colorbar(cs, ax=axis, fraction=.08, pad=0.02, extend=extend)
  if centerlabels and len(clim)>2: cb.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
  elif clim is not None and len(clim)>2: cb.set_ticks( clim )
  axis.set_facecolor(landcolor)
  axis.set_xlim( xLims )
  axis.set_ylim( yLims )
  if annotate:
    axis.annotate('max=%.5g\nmin=%.5g'%(sMax,sMin), xy=(0.0,1.01), xycoords='axes fraction', verticalalignment='bottom', fontsize=10)
    if area is not None:
      axis.annotate('mean=%.5g\nrms=%.5g'%(sMean,sRMS), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
      axis.annotate(' sd=%.5g\n'%(sStd), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='left', fontsize=10)

  if len(xlabel+xunits)>0: axis.set_xlabel(label(xlabel, xunits), fontsize=14)
  if len(ylabel+yunits)>0: axis.set_ylabel(label(ylabel, yunits), fontsize=14)
  if len(title)>0: axis.set_title(title)
  if len(suptitle)>0: plt.suptitle(suptitle)

  if save is not None: plt.savefig(save); plt.close()
  if interactive: addInteractiveCallbacks()
  if show: plt.show(block=False)
  if add_cbar:
    return
  else:
    return cs

def xycompare(field1, field2, x=None, y=None, area=None,
  xlabel=None, xunits=None, ylabel=None, yunits=None,
  title1='', title2='', title3='A - B', addplabel=True, suptitle='',
  clim=None, colormap=None, extend=None, centerlabels=False,
  dlim=None, dcolormap=None, dextend=None, centerdlabels=False,
  nbins=None, landcolor=[.5,.5,.5], sector=None, webversion=False,
  aspect=None, resolution=None, axis=None, npanels=3, sigma=2.,
  ignore=None, save=None, debug=False, show=False, interactive=False):
  """
  Renders n-panel plot of two scalar fields, field1(x,y) and field2(x,y).

  Arguments:
  field1        Scalar 2D array to be plotted and compared to field2.
  field2        Scalar 2D array to be plotted and compared to field1.
  x             x coordinate (1D or 2D array). If x is the same size as field then x is treated as
                the cell center coordinates.
  y             y coordinate (1D or 2D array). If x is the same size as field then y is treated as
                the cell center coordinates.
  area          2D array of cell areas (used for statistics). Default None.
  xlabel        The label for the x axis. Default 'Longitude'.
  xunits        The units for the x axis. Default 'degrees E'.
  ylabel        The label for the y axis. Default 'Latitude'.
  yunits        The units for the y axis. Default 'degrees N'.
  title1        The title to place at the top of panel 1. Default ''.
  title2        The title to place at the top of panel 1. Default ''.
  title3        The title to place at the top of panel 1. Default 'A-B'.
  addplabel     Adds a 'A:' or 'B:' to the title1 and title2. Default True.
  suptitle      The super-title to place at the top of the figure. Default ''.
  sector        Restrcit plot to a specific sector. Default 'None' (i.e. global).
  clim          A tuple of (min,max) color range OR a list of contour levels for the field plots. Default None.
  sigma         Sigma range for difference plot autocolor levels. Default is to span a 2. sigma range
  colormap      The name of the colormap to use for the field plots. Default None.
  extend        Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels  If True, will move the colorbar labels to the middle of the interval. Default False.
  dlim          A tuple of (min,max) color range OR a list of contour levels for the difference plot. Default None.
  dcolormap     The name of the colormap to use for the difference plot. Default None.
  dextend       For the difference colorbar. Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerdlabels If True, will move the difference colorbar labels to the middle of the interval. Default False.
  nbins         The number of colors levels (used is clim is missing or only specifies the color range).
  landcolor     An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  aspect        The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution    The vertical resolution of the figure given in pixels. Default 1280.
  axis          List with axis handles to plot to. Default None.
  npanels       Number of panels to display (1, 2 or 3). Default 3.
  ignore        A value to use as no-data (NaN). Default None.
  save          Name of file to save figure in. Default None.
  debug         If true, report stuff for debugging. Default False.
  show          If true, causes the figure to appear on screen. Used for testing. Default False.
  webversion    If true, set options specific for displaying figures in a web browser. Default False.
  interactive   If true, adds interactive features such as zoom, close and cursor. Default False.
  """

  if (field1.shape)!=(field2.shape): raise Exception('field1 and field2 must be the same shape')

  # Create coordinates if not provided
  xlabel, xunits, ylabel, yunits = createXYlabels(x, y, xlabel, xunits, ylabel, yunits)
  if debug: print('x,y label/units=',xlabel,xunits,ylabel,yunits)
  xCoord, yCoord = createXYcoords(field1, x, y)

  # Establish ranges for sectors
  lonRange, latRange, hspace, titleOffset = sectorRanges(sector=sector)

  # Diagnose statistics
  if sector == None or sector == 'global':
    if ignore is not None: maskedField1 = numpy.ma.masked_array(field1, mask=[field1==ignore])
    else: maskedField1 = field1.copy()
    if ignore is not None: maskedField2 = numpy.ma.masked_array(field2, mask=[field2==ignore])
    else: maskedField2 = field2.copy()
  else:
    maskedField1 = regionalMasking(field1,yCoord,xCoord,latRange,lonRange)
    maskedField2 = regionalMasking(field2,yCoord,xCoord,latRange,lonRange)
    areaCopy = numpy.ma.array(area,mask=maskedField1.mask,copy=True)

  s1Min, s1Max, s1Mean, s1Std, s1RMS = myStats(maskedField1, area, debug=debug)
  s2Min, s2Max, s2Mean, s2Std, s2RMS = myStats(maskedField2, area, debug=debug)
  dMin, dMax, dMean, dStd, dRMS = myStats(maskedField1 - maskedField2, area, debug=debug)
  if s1Mean is not None: dRxy = corr(maskedField1 - s1Mean, maskedField2 - s2Mean, area)
  else: dRxy = None
  s12Min = min(s1Min, s2Min); s12Max = max(s1Max, s2Max)
  xLims = boundaryStats(xCoord); yLims = boundaryStats(yCoord)
  if debug:
    print('s1: min, max, mean =', s1Min, s1Max, s1Mean)
    print('s2: min, max, mean =', s2Min, s2Max, s2Mean)
    print('diff: min, max, mean, std, rms =', dMin, dMax, dMean, dStd, dRMS)
    print('s12: min, max =', s12Min, s12Max)

  # Choose colormap
  if nbins is None and (clim is None or len(clim)==2): cBins=35
  else: cBins=nbins
  if nbins is None and (dlim is None or len(dlim)==2): nbins=35
  if colormap is None: colormap = chooseColorMap(s12Min, s12Max)
  cmap, norm, extend = chooseColorLevels(s12Min, s12Max, colormap, clim=clim, nbins=cBins, extend=extend)

  if addplabel: preTitleA = 'A: '; preTitleB = 'B: '
  else: preTitleA = ''; preTitleB = ''
  if axis is None:
    setFigureSize(aspect, resolution, npanels=npanels, debug=debug)

  if npanels in [2,3]:
    if axis is not None:
      axis1=axis[0]
    else:
      axis1 = plt.subplot(npanels,1,1)
    if sector == None or sector == 'global':
      cs = axis1.pcolormesh(xCoord, yCoord, maskedField1, cmap=cmap, norm=norm)
      if interactive: addStatusBar(xCoord, yCoord, maskedField1)
      cb1 = plt.colorbar(cs,ax=axis1, fraction=.08, pad=0.02, extend=extend)
      axis1.set_xlim( xLims ); axis1.set_ylim( yLims )
      axis1.set_xticklabels([''])
    else:
      plotBasemapPanel(maskedField1, sector, xCoord, yCoord, lonRange, latRange, \
                       cmap, norm, interactive, extend)
    if centerlabels and len(clim)>2: cb1.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
    axis1.set_facecolor(landcolor)
    annotateStats(axis1, s1Min, s1Max, s1Mean, s1Std, s1RMS, webversion=webversion)
    if len(ylabel+yunits)>0: axis1.set_ylabel(label(ylabel, yunits))
    if len(title1)>0:
      if webversion == True: axis1.annotate(preTitleA+title1, xy=(0.5,1.14), xycoords='axes fraction', \
                                           verticalalignment='bottom', horizontalalignment='center', fontsize=12)
      else:
        axis1.set_title(preTitleA+title1)
    if axis is not None:
      axis2=axis[1]
    else:
      axis2 = plt.subplot(npanels,1,2)
    if sector == None or sector == 'global':
      cs = axis2.pcolormesh(xCoord, yCoord, maskedField2, cmap=cmap, norm=norm)
      if interactive: addStatusBar(xCoord, yCoord, maskedField2)
      cb2 = plt.colorbar(cs,ax=axis2,fraction=.08, pad=0.02, extend=extend)
      axis2.set_xlim( xLims ); axis2.set_ylim( yLims )
      if npanels>2: axis2.set_xticklabels([''])
    else:
      plotBasemapPanel(maskedField2, sector, xCoord, yCoord, lonRange, latRange, \
                       cmap, norm, interactive, extend)
    if centerlabels and len(clim)>2: cb2.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
    axis2.set_facecolor(landcolor)
    annotateStats(axis2, s2Min, s2Max, s2Mean, s2Std, s2RMS, webversion=webversion)
    if len(ylabel+yunits)>0: axis2.set_ylabel(label(ylabel, yunits))
    if len(title2)>0:
      if webversion == True: axis2.annotate(preTitleB+title2, xy=(0.5,titleOffset), xycoords='axes fraction', \
                                           verticalalignment='bottom', horizontalalignment='center', fontsize=12)
      else:
        axis2.set_title(preTitleB+title2)

  if npanels in [1,3]:
    if axis is not None:
      axis3=axis[-1]
    else:
      axis3 = plt.subplot(npanels,1,npanels)
    if sector == None or sector == 'global':
      if dcolormap is None: dcolormap = chooseColorMap(dMin, dMax)
      if dlim is None and dStd is not None:
        cmap, norm, dextend = chooseColorLevels(dMean-sigma*dStd, dMean+sigma*dStd, dcolormap, clim=dlim, nbins=nbins, \
                                                extend=dextend, autocenter=True)
      else:
        cmap, norm, dextend = chooseColorLevels(dMin, dMax, dcolormap, clim=dlim, nbins=nbins, extend=dextend, autocenter=True)
      cs = axis3.pcolormesh(xCoord, yCoord, maskedField1 - maskedField2, cmap=cmap, norm=norm)
      if interactive: addStatusBar(xCoord, yCoord, maskedField1 - maskedField2)
      if dextend is None: dextend = extend
      cb3 = plt.colorbar(cs, ax=axis3,fraction=.08, pad=0.02, extend=dextend) # was extend!
      if centerdlabels and len(dlim)>2: cb3.set_ticks(  0.5*(dlim[:-1]+dlim[1:]) )
      axis3.set_facecolor(landcolor)
      axis3.set_xlim( xLims ); axis3.set_ylim( yLims )
      annotateStats(axis3, dMin, dMax, dMean, dStd, dRMS, webversion=webversion)
      if len(ylabel+yunits)>0: axis3.set_ylabel(label(ylabel, yunits))
      if len(title3)>0: axis3.set_title(title3)
    elif sector != None:
      # Copy data array, mask, and compute stats / color levels
      maskedDiffField = numpy.ma.array(maskedField1 - maskedField2)
      dMin, dMax, dMean, dStd, dRMS = myStats(maskedDiffField, areaCopy, debug=debug)
      if dcolormap is None: dcolormap = chooseColorMap(dMin, dMax,difference=True)
      if dlim is None and dStd>0:
        cmap, norm, dextend = chooseColorLevels(dMean-sigma*dStd, dMean+sigma*dStd, dcolormap, clim=dlim, nbins=nbins, \
                                                extend='both', autocenter=True)
      else:
        cmap, norm, dextend = chooseColorLevels(dMin, dMax, dcolormap, clim=dlim, nbins=nbins, extend=dextend, autocenter=True)
      # Set up Basemap projection
      plotBasemapPanel(maskedField1 - maskedField2, sector, xCoord, yCoord, lonRange, latRange, \
                       cmap, norm, interactive, dextend)
      annotateStats(axis3, dMin, dMax, dMean, dStd, dRMS, webversion=webversion)
      if len(ylabel+yunits)>0: axis3.set_ylabel(label(ylabel, yunits))
      if len(title3)>0:
        if webversion == True: axis3.annotate(title3, xy=(0.5,titleOffset), xycoords='axes fraction', verticalalignment='bottom',
                                             horizontalalignment='center', fontsize=12)
        else:
          axis3.set_title(title3)
    else:
      raise ValueError('Invalid sector specified')

  if webversion: plt.subplots_adjust(hspace=hspace)
  if webversion == True:
    fig = plt.gcf()
    fig.text(0.5,0.02,'Generated by m6plot on dora.gfdl.noaa.gov',fontsize=9,horizontalalignment='center')

  plt.suptitle(suptitle,y=1.0)

  if save is not None: plt.savefig(save,bbox_inches='tight')
  if interactive: addInteractiveCallbacks()
  if show: plt.show(block=False)

  return

def annotateStats(axis, sMin, sMax, sMean, sStd, sRMS, webversion=False):
  '''
  Adds (annotates) stats in figures.
  '''
  if webversion == True: fontsize=9
  else: fontsize=10
  axis.annotate('max=%.5g\nmin=%.5g'%(sMax,sMin), xy=(0.0,1.025), xycoords='axes fraction', \
                  verticalalignment='bottom', fontsize=fontsize)
  if sMean is not None:
    axis.annotate('mean=%.5g\nrms=%.5g'%(sMean,sRMS), xy=(1.0,1.025), xycoords='axes fraction', \
                  verticalalignment='bottom', horizontalalignment='right', fontsize=fontsize)
    axis.annotate(' sd=%.5g\n'%(sStd), xy=(1.0,1.025), xycoords='axes fraction', verticalalignment='bottom', \
                    horizontalalignment='left', fontsize=fontsize)

  return

def yzplot(field, y=None, z=None,
  ylabel=None, yunits=None, zlabel=None, zunits=None,
  splitscale=None,
  title='', suptitle='',
  clim=None, colormap=None, extend=None, centerlabels=False,
  nbins=None, landcolor=[.5,.5,.5], show_stats=2,
  aspect=[16,9], resolution=576, axis=None,
  ignore=None, save=None, debug=False, show=False, interactive=False):
  """
  Renders section plot of scalar field, field(x,z).

  Arguments:
  field       Scalar 2D array to be plotted.
  y           y (or x) coordinate (1D array). If y is the same size as field then x is treated as
              the cell center coordinates.
  z           z coordinate (1D or 2D array). If z is the same size as field then y is treated as
              the cell center coordinates.
  ylabel      The label for the x axis. Default 'Latitude'.
  yunits      The units for the x axis. Default 'degrees N'.
  zlabel      The label for the z axis. Default 'Elevation'.
  zunits      The units for the z axis. Default 'm'.
  splitscale    A list of depths to define equal regions of projection in the vertical, e.g. [0.,-1000,-6500]
  title       The title to place at the top of the panel. Default ''.
  suptitle    The super-title to place at the top of the figure. Default ''.
  clim        A tuple of (min,max) color range OR a list of contour levels. Default None.
  colormap    The name of the colormap to use. Default None.
  extend      Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels If True, will move the colorbar labels to the middle of the interval. Default False.
  nbins       The number of colors levels (used is clim is missing or only specifies the color range).
  landcolor   An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  aspect      The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution  The vertical resolution of the figure given in pixels. Default 720.
  show_stats  Integer. If = 2 (default), annotate min, max, mean, std and rms in the figure. If = 1, just min and max.
              If = 0, no stats are annotated.
  axis         The axis handle to plot to. Default None.
  ignore      A value to use as no-data (NaN). Default None.
  save        Name of file to save figure in. Default None.
  debug       If true, report stuff for debugging. Default False.
  show        If true, causes the figure to appear on screen. Used for testing. Default False.
  interactive If true, adds interactive features such as zoom, close and cursor. Default False.
  """

  # Create coordinates if not provided
  ylabel, yunits, zlabel, zunits = createYZlabels(y, z, ylabel, yunits, zlabel, zunits)
  if debug: print('y,z label/units=',ylabel,yunits,zlabel,zunits)
  if len(y)==z.shape[-1]: y = expand(y)
  elif len(y)==z.shape[-1]+1: y = y
  else: raise Exception('Length of y coordinate should be equal or 1 longer than horizontal length of z')
  if ignore is not None: maskedField = numpy.ma.masked_array(field, mask=[field==ignore])
  else: maskedField = field.copy()
  yCoord, zCoord, field2 = m6toolbox.section2quadmesh(y, z, maskedField)

  # Diagnose statistics
  sMin, sMax, sMean, sStd, sRMS = myStats(maskedField, yzWeight(y, z), debug=debug)
  yLims = numpy.amin(yCoord), numpy.amax(yCoord)
  zLims = boundaryStats(zCoord)

  # Choose colormap
  if nbins is None and (clim is None or len(clim)==2): nbins=35
  if colormap is None: colormap = chooseColorMap(sMin, sMax)
  cmap, norm, extend = chooseColorLevels(sMin, sMax, colormap, clim=clim, nbins=nbins, extend=extend)

  if axis is None:
    setFigureSize(aspect, resolution, debug=debug)
    #plt.gcf().subplots_adjust(left=.10, right=.99, wspace=0, bottom=.09, top=.9, hspace=0)
    axis = plt.gca()

  cs = axis.pcolormesh(yCoord, zCoord, field2, cmap=cmap, norm=norm)
  if interactive: addStatusBar(yCoord, zCoord, field2)
  cb = plt.colorbar(cs,ax=axis, fraction=.08, pad=0.02, extend=extend)
  if centerlabels and len(clim)>2: cb.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
  axis.set_facecolor(landcolor)
  if splitscale is not None:
    for zzz in splitscale[1:-1]: axis.axhline(zzz,color='k',linestyle='--')
    axis.set_yscale('splitscale', zval=splitscale)
  axis.set_xlim( yLims ); axis.set_ylim( zLims )
  if show_stats > 0:
    axis.annotate('max=%.5g\nmin=%.5g'%(sMax,sMin), xy=(0.0,1.01), xycoords='axes fraction', verticalalignment='bottom', fontsize=10)
    if show_stats > 1:
      if sMean is not None:
        axis.annotate('mean=%.5g\nrms=%.5g'%(sMean,sRMS), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
        axis.annotate(' sd=%.5g\n'%(sStd), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='left', fontsize=10)

  if len(ylabel+yunits)>0: axis.set_xlabel(label(ylabel, yunits))
  if len(zlabel+zunits)>0: axis.set_ylabel(label(zlabel, zunits))
  if len(title)>0: axis.set_title(title)
  if len(suptitle)>0: plt.suptitle(suptitle)

  if save is not None: plt.savefig(save)
  if interactive: addInteractiveCallbacks()
  if show: plt.show(block=False)

def yzcompare(field1, field2, y=None, z=None,
  ylabel=None, yunits=None, zlabel=None, zunits=None,
  splitscale=None,
  title1='', title2='', title3='A - B', addplabel=True, suptitle='',
  clim=None, colormap=None, extend=None, centerlabels=False,
  dlim=None, dcolormap=None, dextend=None, centerdlabels=False,
  nbins=None, landcolor=[.5,.5,.5], sigma=2., webversion=False,
  aspect=None, resolution=None, axis=None, npanels=3, contour=False,
  ignore=None, save=None, debug=False, show=False, interactive=False):
  """
  Renders n-panel plot of two scalar fields, field1(x,y) and field2(x,y).

  Arguments:
  field1        Scalar 2D array to be plotted and compared to field2.
  field2        Scalar 2D array to be plotted and compared to field1.
  y             y coordinate (1D array). If y is the same size as field then y is treated as
                the cell center coordinates.
  z             z coordinate (1D or 2D array). If z is the same size as field then z is treated as
                the cell center coordinates.
  ylabel        The label for the y axis. Default 'Latitude'.
  yunits        The units for the y axis. Default 'degrees N'.
  zlabel        The label for the z axis. Default 'Elevation'.
  zunits        The units for the z axis. Default 'm'.
  splitscale    A list of depths to define equal regions of projection in the vertical, e.g. [0.,-1000,-6500]
  title1        The title to place at the top of panel 1. Default ''.
  title2        The title to place at the top of panel 1. Default ''.
  title3        The title to place at the top of panel 1. Default 'A-B'.
  addplabel     Adds a 'A:' or 'B:' to the title1 and title2. Default True.
  suptitle      The super-title to place at the top of the figure. Default ''.
  clim          A tuple of (min,max) color range OR a list of contour levels for the field plots. Default None.
  sigma         Sigma range for difference plot autocolor levels. Default is to span a 2. sigma range
  colormap      The name of the colormap to use for the field plots. Default None.
  extend        Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels  If True, will move the colorbar labels to the middle of the interval. Default False.
  dlim          A tuple of (min,max) color range OR a list of contour levels for the difference plot. Default None.
  dcolormap     The name of the colormap to use for the differece plot. Default None.
  dextend       For the difference colorbar. Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerdlabels If True, will move the difference colorbar labels to the middle of the interval. Default False.
  nbins         The number of colors levels (used is clim is missing or only specifies the color range).
  landcolor     An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  aspect        The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution    The vertical resolution of the figure given in pixels. Default 1280.
  axis          The axis handle to plot to. Default None.
  npanels       Number of panels to display (1, 2 or 3). Default 3.
  contour       If true, draw and label contour lines. Default is False.
  ignore        A value to use as no-data (NaN). Default None.
  save          Name of file to save figure in. Default None.
  debug         If true, report stuff for debugging. Default False.
  show          If true, causes the figure to appear on screen. Used for testing. Default False.
  webversion    If true, set options specific for displaying figures in a web browser. Default False.
  interactive   If true, adds interactive features such as zoom, close and cursor. Default False.
  """

  if (field1.shape)!=(field2.shape): raise Exception('field1 and field2 must be the same shape')

  # Create coordinates if not provided
  ylabel, yunits, zlabel, zunits = createYZlabels(y, z, ylabel, yunits, zlabel, zunits)
  if debug: print('y,z label/units=',ylabel,yunits,zlabel,zunits)
  if len(y)==z.shape[-1]: y= expand(y)
  elif len(y)==z.shape[-1]+1: y= y
  else: raise Exception('Length of y coordinate should be equal or 1 longer than horizontal length of z')
  if ignore is not None: maskedField1 = numpy.ma.masked_array(field1, mask=[field1==ignore])
  else: maskedField1 = field1.copy()
  yCoord, zCoord, field1 = m6toolbox.section2quadmesh(y, z, maskedField1)

  # Diagnose statistics
  yzWeighting = yzWeight(y, z)
  s1Min, s1Max, s1Mean, s1Std, s1RMS = myStats(maskedField1, yzWeighting, debug=debug)
  if ignore is not None: maskedField2 = numpy.ma.masked_array(field2, mask=[field2==ignore])
  else: maskedField2 = field2.copy()
  yCoord, zCoord, field2 = m6toolbox.section2quadmesh(y, z, maskedField2)
  s2Min, s2Max, s2Mean, s2Std, s2RMS = myStats(maskedField2, yzWeighting, debug=debug)
  dMin, dMax, dMean, dStd, dRMS = myStats(maskedField1 - maskedField2, yzWeighting, debug=debug)
  dRxy = corr(maskedField1 - s1Mean, maskedField2 - s2Mean, yzWeighting)
  s12Min = min(s1Min, s2Min); s12Max = max(s1Max, s2Max)
  xLims = numpy.amin(yCoord), numpy.amax(yCoord); yLims = boundaryStats(zCoord)
  if debug:
    print('s1: min, max, mean =', s1Min, s1Max, s1Mean)
    print('s2: min, max, mean =', s2Min, s2Max, s2Mean)
    print('s12: min, max =', s12Min, s12Max)

  # Choose colormap
  if nbins is None and (clim is None or len(clim)==2): cBins=35
  else: cBins=nbins
  if nbins is None and (dlim is None or len(dlim)==2): nbins=35
  if colormap is None: colormap = chooseColorMap(s12Min, s12Max)
  cmap, norm, extend = chooseColorLevels(s12Min, s12Max, colormap, clim=clim, nbins=cBins, extend=extend)

  if addplabel: preTitleA = 'A: '; preTitleB = 'B: '
  else: preTitleA = ''; preTitleB = ''

  if axis is None:
    setFigureSize(aspect, resolution, npanels=npanels, debug=debug)
    #plt.gcf().subplots_adjust(left=.13, right=.94, wspace=0, bottom=.05, top=.94, hspace=0.15)

  if npanels in [2, 3]:
    axis = plt.subplot(npanels,1,1)
    plt.pcolormesh(yCoord, zCoord, field1, cmap=cmap, norm=norm)
    if interactive: addStatusBar(yCoord, zCoord, field1)
    cb1 = plt.colorbar(fraction=.08, pad=0.02, extend=extend)
    if centerlabels and len(clim)>2: cb1.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
    axis.set_facecolor(landcolor)
    if contour:
      zz = 0.5 * ( z[:-1] + z[1:])
      yy = 0.5 * ( y[:-1] + y[1:])
      print(yy.shape, zz.shape, maskedField1.shape)
      matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
      cs = plt.contour(yy+0*zz, zz, maskedField1, colors='k')
      plt.clabel(cs, fmt='%2.1f', fontsize=14)

    if splitscale is not None:
      for zzz in splitscale[1:-1]: plt.axhline(zzz,color='k',linestyle='--')
      axis.set_yscale('splitscale', zval=splitscale)
    plt.xlim( xLims ); plt.ylim( yLims )
    annotateStats(axis, s1Min, s1Max, s1Mean, s1Std, s1RMS, webversion=webversion)
    axis.set_xticklabels([''])
    if len(zlabel+zunits)>0: plt.ylabel(label(zlabel, zunits))
    if len(title1)>0: plt.title(preTitleA+title1)

    axis = plt.subplot(npanels,1,2)
    plt.pcolormesh(yCoord, zCoord, field2, cmap=cmap, norm=norm)
    if interactive: addStatusBar(yCoord, zCoord, field2)
    cb2 = plt.colorbar(fraction=.08, pad=0.02, extend=extend)
    if contour:
      cs = plt.contour(yy+0*zz, zz, maskedField2, colors='k')
      plt.clabel(cs, fmt='%2.1f', fontsize=14)

    if centerlabels and len(clim)>2: cb2.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
    axis.set_facecolor(landcolor)
    if splitscale is not None:
      for zzz in splitscale[1:-1]: plt.axhline(zzz,color='k',linestyle='--')
      axis.set_yscale('splitscale', zval=splitscale)
    plt.xlim( xLims ); plt.ylim( yLims )
    annotateStats(axis, s2Min, s2Max, s2Mean, s2Std, s2RMS, webversion=webversion)
    if npanels>2: axis.set_xticklabels([''])
    if len(zlabel+zunits)>0: plt.ylabel(label(zlabel, zunits))
    if len(title2)>0: plt.title(preTitleB+title2)

  if npanels in [1, 3]:
    axis = plt.subplot(npanels,1,npanels)

    if dcolormap is None: dcolormap = chooseColorMap(dMin, dMax)
    if dlim is None and dStd>0:
      cmap, norm, dextend = chooseColorLevels(dMean-sigma*dStd, dMean+sigma*dStd, dcolormap, clim=dlim, nbins=nbins, extend='both', autocenter=True)
    else:
      cmap, norm, dextend = chooseColorLevels(dMin, dMax, dcolormap, clim=dlim, nbins=nbins, extend=dextend, autocenter=True)
    plt.pcolormesh(yCoord, zCoord, field1 - field2, cmap=cmap, norm=norm)
    if interactive: addStatusBar(yCoord, zCoord, field1 - field2)
    cb3 = plt.colorbar(fraction=.08, pad=0.02, extend=dextend)
    if centerdlabels and len(dlim)>2: cb3.set_ticks(  0.5*(dlim[:-1]+dlim[1:]) )
    axis.set_facecolor(landcolor)
    if splitscale is not None:
      for zzz in splitscale[1:-1]: plt.axhline(zzz,color='k',linestyle='--')
      axis.set_yscale('splitscale', zval=splitscale)
    plt.xlim( xLims ); plt.ylim( yLims )
    annotateStats(axis, dMin, dMax, dMean, dStd, dRMS)
    if len(zlabel+zunits)>0: plt.ylabel(label(zlabel, zunits))

  axis.annotate(' r(A,B)=%.5g\n'%(dRxy), xy=(1.0,-1.07), xycoords='axes fraction', verticalalignment='top', horizontalalignment='center', fontsize=10)
  if len(ylabel+yunits)>0: plt.xlabel(label(ylabel, yunits))
  if len(title3)>0: plt.title(title3)
  if len(suptitle)>0: plt.suptitle(suptitle)

  if save is not None: plt.savefig(save)
  if interactive: addInteractiveCallbacks()
  if show: plt.show(block=False)

def ztplot(field, t=None, z=None,
  tlabel=None, tunits=None, zlabel=None, zunits=None,
  splitscale=None,
  title='', suptitle='', autocenter=False,
  clim=None, colormap=None, extend=None, centerlabels=False,
  nbins=None, landcolor=[.5,.5,.5], contour=False, clevs=[],
  aspect=[16,9], resolution=576, axis=None, add_cbar=True,
  ignore=None, save=None, debug=False, show=False, interactive=False):
  """
  Renders section plot of scalar field, field(x,z).

  Arguments:
  field       Scalar 2D array to be plotted.
  t           t (or time) coordinate (1D array).
  z           z coordinate (1D array).
  tlabel      The label for the t axis. Default 'Time'.
  tunits      The units for the t axis. Default 'Years'.
  zlabel      The label for the z axis. Default 'Elevation'.
  zunits      The units for the z axis. Default 'm'.
  splitscale    A list of depths to define equal regions of projection in the vertical, e.g. [0.,-1000,-6500]
  title       The title to place at the top of the panel. Default ''.
  suptitle    The super-title to place at the top of the figure. Default ''.
  autocenter  If clim generated by script, set to be centered on zero.  Default False.
  add_cbar     If true, the colorbar is added to the panel. Otherwise, the function will return the color
               scale object that be used to create a colorbar. Default is True.
  clim        A tuple of (min,max) color range OR a list of contour levels. Default None.
  colormap    The name of the colormap to use. Default None.
  extend      Can be one of 'both', 'neither', 'max', 'min'. Default None.
  centerlabels If True, will move the colorbar labels to the middle of the interval. Default False.
  nbins       The number of colors levels (used is clim is missing or only specifies the color range).
  landcolor   An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  contour     If true, draw and label contour lines. Default is False.
  clevs       If contour=True, these are the levels to be contoured. Default is [].
  aspect      The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution  The vertical resolution of the figure given in pixels. Default 720.
  axis         The axis handle to plot to. Default None.
  ignore      A value to use as no-data (NaN). Default None.
  save        Name of file to save figure in. Default None.
  debug       If true, report stuff for debugging. Default False.
  show        If true, causes the figure to appear on screen. Used for testing. Default False.
  interactive If true, adds interactive features such as zoom, close and cursor. Default False.
  """

  # Create coordinates if not provided
  tlabel, tunits, zlabel, zunits = createTZlabels(t, z, tlabel, tunits, zlabel, zunits)
  if debug: print('t,z label/units=',tlabel,tunits,zlabel,zunits)
  if ignore is not None: maskedField = numpy.ma.masked_array(field, mask=[field==ignore])
  else: maskedField = field.copy()
  field2 = maskedField.T
  tCoord = t; zCoord = z

  # Diagnose statistics
  sMin, sMax, sMean, sStd, sRMS = myStats(maskedField, None, debug=debug)
  tLims = numpy.amin(tCoord), numpy.amax(tCoord)
  zLims = numpy.amin(zCoord), numpy.amax(zCoord)
  #zLims = boundaryStats(zCoord)

  # Choose colormap
  if nbins is None and (clim is None or len(clim)==2): nbins=35
  if colormap is None: colormap = chooseColorMap(sMin, sMax)
  cmap, norm, extend = chooseColorLevels(sMin, sMax, colormap, clim=clim, nbins=nbins, extend=extend, autocenter=autocenter)

  if axis is None:
    setFigureSize(aspect, resolution, debug=debug)
    #plt.gcf().subplots_adjust(left=.10, right=.99, wspace=0, bottom=.09, top=.9, hspace=0)
    axis = plt.gca()

  cs = axis.pcolormesh(tCoord, zCoord, field2, cmap=cmap, norm=norm)
  if interactive: addStatusBar(tCoord, zCoord, field2)
  if add_cbar: cb = plt.colorbar(cs, ax=axis, fraction=.08, pad=0.02, extend=extend)
  if contour:
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    if len(clevs) > 0:
      cs1 = axis.contour(tCoord, zCoord,field2,clevs,colors='k',lw=0.50)
    else:
      cs1 = axis.contour(tCoord, zCoord,field2,colors='k',lw=0.50)
    axis.clabel(cs1,inline=1, fontsize=10)

  if centerlabels and add_cbar and len(clim)>2: cb.set_ticks(  0.5*(clim[:-1]+clim[1:]) )
  axis.set_facecolor(landcolor)
  if splitscale is not None:
    for zzz in splitscale[1:-1]: axis.axhline(zzz,color='k',linestyle='--')
    axis.set_yscale('splitscale', zval=splitscale)

  axis.set_xlim( tLims ); axis.set_ylim( zLims )
  axis.annotate('max=%.5g\nmin=%.5g'%(sMax,sMin), xy=(0.0,1.01), xycoords='axes fraction', verticalalignment='bottom', fontsize=12)
  if sMean is not None:
    axis.annotate('mean=%.5g\nrms=%.5g'%(sMean,sRMS), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='right', fontsize=12)
    axis.annotate(' sd=%.5g\n'%(sStd), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='left', fontsize=12)
  if len(tlabel+tunits)>0: axis.set_xlabel(label(tlabel, tunits), fontsize=16)
  if len(zlabel+zunits)>0: axis.set_ylabel(label(zlabel, zunits), fontsize=16)
  if len(title)>0: axis.set_title(title)
  if len(suptitle)>0: plt.suptitle(suptitle)
  if splitscale is not None:
    axis.invert_yaxis()
  if save is not None: plt.savefig(save)
  if interactive: addInteractiveCallbacks()
  if show: plt.show(block=False)
  if add_cbar:
    return
  else:
    return cs

def chooseColorMap(sMin, sMax, difference=None):
  """
  Based on the min/max extremes of the data, choose a colormap that fits the data.
  """
  if difference == True: return 'dunnePM'
  elif sMin<0 and sMax>0: return 'dunnePM'
  #elif sMax>0 and sMin<0.1*sMax: return 'hot'
  #elif sMin<0 and sMax>0.1*sMin: return 'hot_r'
  else: return 'dunneRainbow'


def chooseColorLevels(sMin, sMax, colorMapName, clim=None, nbins=None, steps=[1,2,2.5,5,10], extend=None, logscale=False, autocenter=False):
  """
  If nbins is a positive integer, choose sensible color levels with nbins colors.
  If clim is a 2-element tuple, create color levels within the clim range
  or if clim is a vector, use clim as contour levels.
  If clim provides more than 2 color interfaces, nbins must be absent.
  If clim is absent, the sMin,sMax are used as the color range bounds.
  If autocenter is True and clim is None then the automatic color levels are centered.

  Returns cmap, norm and extend.
  """
  if nbins is None and clim is None: raise Exception('At least one of clim or nbins is required.')
  if clim is not None:
    if len(clim)<2: raise Exception('clim must be at least 2 values long.')
    if nbins is None and len(clim)==2: raise Exception('nbins must be provided when clims specifies a color range.')
    if nbins is not None and len(clim)>2: raise Exception('nbins cannot be provided when clims specifies color levels.')
  if clim is None:
    if autocenter:
      levels = MaxNLocator(nbins=nbins, steps=steps).tick_values(min(sMin,-sMax), max(sMax,-sMin))
    else:
      levels = MaxNLocator(nbins=nbins, steps=steps).tick_values(sMin, sMax)
  elif len(clim)==2: levels = MaxNLocator(nbins=nbins, steps=steps).tick_values(clim[0], clim[1])
  else: levels = clim

  nColors = len(levels)-1
  if extend is None:
    if sMin<levels[0] and sMax>levels[-1]: extend = 'both'#; eColors=[1,1]
    elif sMin<levels[0] and sMax<=levels[-1]: extend = 'min'#; eColors=[1,0]
    elif sMin>=levels[0] and sMax>levels[-1]: extend = 'max'#; eColors=[0,1]
    else: extend = 'neither'#; eColors=[0,0]
  eColors = [0,0]
  if extend in ['both', 'min']: eColors[0] = 1
  if extend in ['both', 'max']: eColors[1] = 1

  cmap = plt.get_cmap(colorMapName)#,lut=nColors+eColors[0]+eColors[1])
  #cmap0 = cmap(0.)
  #cmap1 = cmap(1.)
  #cmap = ListedColormap(cmap(range(eColors[0],nColors+1-eColors[1]+eColors[0])))#, N=nColors)
  #if eColors[0]>0: cmap.set_under(cmap0)
  #if eColors[1]>0: cmap.set_over(cmap1)
  if logscale: norm = LogNorm(vmin=levels[0], vmax=levels[-1])
  else: norm = BoundaryNorm(levels, ncolors=cmap.N)
  return cmap, norm, extend


def linCI(min, max, ci, *args):
  """
  Returns list of linearly spaced contour intervals from min to max with spacing ci.
  Unline numpy.arange this max is included IF max = min + ci*N for an integer N.
  """
  if len(args): return numpy.concatenate( (numpy.arange(min, max+ci, ci), linCI(*args)) )
  return numpy.arange(min, max+ci, ci)


def pmCI(min, max, ci, *args):
  """
  Returns list of linearly spaced contour intervals from -max to -min then min to max with spacing ci.
  Unline numpy.arange this max is included IF max = min + ci*N for an integer N.
  """
  ci = linCI(min, max, ci, *args)
  if ci[0]>0: return numpy.concatenate( (-ci[::-1],ci) )
  else: return numpy.concatenate( (-ci[::-1],ci[1:]) )

def myStats(s, area, debug=False):
  """
  Calculates mean, standard deviation and root-mean-square of s.
  """
  sMin = numpy.ma.min(s); sMax = numpy.ma.max(s)
  if debug: print('myStats: min(s) =',sMin)
  if debug: print('myStats: max(s) =',sMax)
  if area is None: return sMin, sMax, None, None, None
  weight = area.copy()
  if debug: print('myStats: sum(area) =',numpy.ma.sum(weight))
  if not numpy.ma.getmask(s).any()==numpy.ma.nomask: weight[s.mask] = 0.
  sumArea = numpy.ma.sum(weight)
  if debug: print('myStats: sum(area) =',sumArea,'after masking')
  if debug: print('myStats: sum(s) =',numpy.ma.sum(s))
  if debug: print('myStats: sum(area*s) =',numpy.ma.sum(weight*s))
  mean = numpy.ma.sum(weight*s)/sumArea
  std = math.sqrt( numpy.ma.sum( weight*((s-mean)**2) )/sumArea )
  rms = math.sqrt( numpy.ma.sum( weight*(s**2) )/sumArea )
  if debug: print('myStats: mean(s) =',mean)
  if debug: print('myStats: std(s) =',std)
  if debug: print('myStats: rms(s) =',rms)
  return sMin, sMax, mean, std, rms


def corr(s1, s2, area):
  """
  Calculates the correlation coefficient between s1 and s2, assuming s1 and s2 have
  not mean. That is s1 = S - mean(S), etc.
  """
  weight = area.copy()
  if not numpy.ma.getmask(s1).any()==numpy.ma.nomask: weight[s1.mask] = 0.
  sumArea = numpy.ma.sum(weight)
  v1 = numpy.ma.sum( weight*(s1**2) )/sumArea
  v2 = numpy.ma.sum( weight*(s2**2) )/sumArea
  if v1==0 or v2==0: return numpy.NaN
  rxy = numpy.ma.sum( weight*(s1*s2) )/sumArea / math.sqrt( v1*v2 )
  return rxy


def createXYcoords(s, x, y):
  """
  Checks that x and y are appropriate 2D corner coordinates
  and tries to make some if they are not.
  """
  nj, ni = s.shape
  if x is None: xCoord = numpy.arange(0., ni+1)
  else: xCoord = numpy.ma.filled(x, 0.)
  if y is None: yCoord = numpy.arange(0., nj+1)
  else: yCoord = numpy.ma.filled(y, 0.)

  # Turn coordinates into 2D arrays if 1D arrays were provided
  if len(xCoord.shape)==1:
    nxy = yCoord.shape
    xCoord = numpy.matlib.repmat(xCoord, nxy[0], 1)
  nxy = xCoord.shape
  if len(yCoord.shape)==1: yCoord = numpy.matlib.repmat(yCoord.T, nxy[-1], 1).T
  if xCoord.shape!=yCoord.shape: raise Exception('The shape of coordinates are mismatched!')

  # Create corner coordinates from center coordinates is center coordinates were provided
  if xCoord.shape!=yCoord.shape: raise Exception('The shape of coordinates are mismatched!')
  if s.shape==xCoord.shape:
    xCoord = expandJ( expandI( xCoord ) )
    yCoord = expandJ( expandI( yCoord ) )
  return xCoord, yCoord


def expandI(a):
  """
  Expands an array by one column, averaging the data to the middle columns and
  extrapolating for the first and last columns. Needed for shifting coordinates
  from centers to corners.
  """
  nj, ni = a.shape
  b = numpy.zeros((nj, ni+1))
  b[:,1:-1] = 0.5*( a[:,:-1] + a[:,1:] )
  b[:,0] = a[:,0] + 0.5*( a[:,0] - a[:,1] )
  b[:,-1] = a[:,-1] + 0.5*( a[:,-1] - a[:,-2] )
  return b


def expandJ(a):
  """
  Expands an array by one row, averaging the data to the middle columns and
  extrapolating for the first and last rows. Needed for shifting coordinates
  from centers to corners.
  """
  nj, ni = a.shape
  b = numpy.zeros((nj+1, ni))
  b[1:-1,:] = 0.5*( a[:-1,:] + a[1:,:] )
  b[0,:] = a[0,:] + 0.5*( a[0,:] - a[1,:] )
  b[-1,:] = a[-1,:] + 0.5*( a[-1,:] - a[-2,:] )
  return b


def expand(a):
  """
  Expands a vector by one element, averaging the data to the middle columns and
  extrapolating for the first and last rows. Needed for shifting coordinates
  from centers to corners.
  """
  b = numpy.zeros((len(a)+1))
  b[1:-1] = 0.5*( a[:-1] + a[1:] )
  b[0] = a[0] + 0.5*( a[0] - a[1] )
  b[-1] = a[-1] + 0.5*( a[-1] - a[-2] )
  return b


def boundaryStats(a):
  """
  Returns the minimum and maximum values of a only on the boundaries of the array.
  """
  amin = numpy.amin(a[0,:])
  amin = min(amin, numpy.amin(a[1:,-1]))
  amin = min(amin, numpy.amin(a[-1,:-1]))
  amin = min(amin, numpy.amin(a[1:-1,0]))
  amax = numpy.amax(a[0,:])
  amax = max(amax, numpy.amax(a[1:,-1]))
  amax = max(amax, numpy.amax(a[-1,:-1]))
  amax = max(amax, numpy.amax(a[1:-1,0]))
  return amin, amax


def setFigureSize(aspect=None, verticalresolution=None, horiztonalresolution=None,
  npanels=1, debug=False):
  """
  Set the figure size based on vertical resolution and aspect ratio (tuple of W,H).
  """
  if (not horiztonalresolution is None) and (not verticalresolution is None):
    if aspect is None: aspect=[horiztonalresolution, verticalresolution]
    else: raise Exception('Aspect-ratio and both h-/v- resolutions can not be specified together')
  if aspect is None: aspect = {1:[16,9], 2:[1,1], 3:[7,10]}[npanels]
  if (not horiztonalresolution is None) and (verticalresolution is None):
    verticalresolution = int(1.*aspect[1]/aspect[0] * horiztonalresolution)
  if verticalresolution is None: verticalresolution = {1:576, 2:720, 3:1200}[npanels]
  width = int(1.*aspect[0]/aspect[1] * verticalresolution) # First guess
  if debug: print('setFigureSize: first guess width =',width)
  width = width + ( width % 2 ) # Make even
  if debug: print('setFigureSize: corrected width =',width)
  if debug: print('setFigureSize: height =',verticalresolution)
  fig = plt.figure(figsize=(width/100., verticalresolution/100.)) # 100 dpi always?
  if npanels==1: plt.gcf().subplots_adjust(left=.08, right=.99, wspace=0, bottom=.09, top=.9, hspace=0)
  elif npanels==2: plt.gcf().subplots_adjust(left=.11, right=.94, wspace=0, bottom=.09, top=.9, hspace=0.15)
  elif npanels==3: plt.gcf().subplots_adjust(left=.11, right=.94, wspace=0, bottom=.05, top=.93, hspace=0.15)
  elif npanels==0: pass
  else: raise Exception('npanels out of range')
  return fig

def label(label, units):
  """
  Combines a label string and units string together in the form 'label [units]'
  unless one of the other is empty.
  """
  string = r''+label
  if len(units)>0: string = string + ' [' + units + ']'
  return string


def createXYlabels(x, y, xlabel, xunits, ylabel, yunits):
  """
  Checks that x and y labels are appropriate and tries to make some if they are not.
  """
  if x is None:
    if xlabel is None: xlabel='i'
    if xunits is None: xunits=''
  else:
    if xlabel is None: xlabel='Longitude'
    #if xunits is None: xunits=u'\u00B0E'
    if xunits is None: xunits=r'$\degree$E'
  if y is None:
    if ylabel is None: ylabel='j'
    if yunits is None: yunits=''
  else:
    if ylabel is None: ylabel='Latitude'
    #if yunits is None: yunits=u'\u00B0N'
    if yunits is None: yunits=r'$\degree$N'
  return xlabel, xunits, ylabel, yunits


def addInteractiveCallbacks():
  """
  Adds interactive features to a plot on screen.
  Key 'q' to close window.
  Zoom button to center.
  Zoom wheel to zoom in and out.
  """
  def keyPress(event):
    if event.key=='Q': exit(0) # Exit python
    elif event.key=='q': plt.close() # Close just the active figure
  class hiddenStore:
    def __init__(self,axis):
      self.axis = axis
      self.xMin, self.xMax = axis.get_xlim()
      self.yMin, self.yMax = axis.get_ylim()
  save = hiddenStore(plt.gca())
  def zoom(event): # Scroll wheel up/down
    if event.button == 'up': scaleFactor = 1/1.5 # deal with zoom in
    elif event.button == 'down': scaleFactor = 1.5 # deal with zoom out
    elif event.button == 2: scaleFactor = 1.0
    else: return
    axis = event.inaxes
    axmin,axmax=axis.get_xlim(); aymin,aymax=axis.get_ylim();
    (axmin,axmax),(aymin,aymax) = newLims(
        (axmin,axmax), (aymin,aymax), (event.xdata, event.ydata),
        (save.xMin,save.xMax), (save.yMin,save.yMax), scaleFactor)
    if axmin is None: return
    for axis in plt.gcf().get_axes():
      if axis.get_navigate():
        axis.set_xlim(axmin, axmax); axis.set_ylim(aymin, aymax)
    plt.draw() # force re-draw
  def zoom2(event): zoom(event)
  plt.gcf().canvas.mpl_connect('key_press_event', keyPress)
  plt.gcf().canvas.mpl_connect('scroll_event', zoom)
  plt.gcf().canvas.mpl_connect('button_press_event', zoom2)


def addStatusBar(xCoord, yCoord, zData):
  """
  Reformats status bar message
  """
  class hiddenStore:
    def __init__(self,axis):
      self.axis = axis
      self.xMin, self.xMax = axis.get_xlim()
      self.yMin, self.yMax = axis.get_ylim()
  save = hiddenStore(plt.gca())
  def statusMessage(x,y):
    # THIS NEEDS TESTING FOR ACCURACY, ESPECIALLY IN YZ PLOTS -AJA
    if len(xCoord.shape)==1 and len(yCoord.shape)==1:
      # -2 needed because of coords are for vertices and need to be averaged to centers
      i = min(range(len(xCoord)-2), key=lambda l: abs((xCoord[l]+xCoord[l+1])/2.-x))
      j = min(range(len(yCoord)-2), key=lambda l: abs((yCoord[l]+yCoord[l+1])/2.-y))
    elif len(xCoord.shape)==1 and len(yCoord.shape)==2:
      i = min(range(len(xCoord)-2), key=lambda l: abs((xCoord[l]+xCoord[l+1])/2.-x))
      j = min(range(len(yCoord[:,i])-1), key=lambda l: abs((yCoord[l,i]+yCoord[l+1,i])/2.-y))
    elif len(xCoord.shape)==2 and len(yCoord.shape)==2:
      idx = numpy.abs( numpy.fabs( xCoord[0:-1,0:-1]+xCoord[1:,1:]+xCoord[0:-1,1:]+xCoord[1:,0:-1]-4*x)
          +numpy.fabs( yCoord[0:-1,0:-1]+yCoord[1:,1:]+yCoord[0:-1,1:]+yCoord[1:,0:-1]-4*y) ).argmin()
      j,i = numpy.unravel_index(idx,zData.shape)
    else: raise Exception('Combindation of coordinates shapes is VERY UNUSUAL!')
    if not i is None:
      val = zData[j,i]
      if val is numpy.ma.masked: return 'x,y=%.3f,%.3f  f(%i,%i)=NaN'%(x,y,i+1,j+1)
      else: return 'x,y=%.3f,%.3f  f(%i,%i)=%g'%(x,y,i+1,j+1,val)
    else: return 'x,y=%.3f,%.3f'%(x,y)
  plt.gca().format_coord = statusMessage


def newLims(cur_xlim, cur_ylim, cursor, xlim, ylim, scale_factor):
  cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
  cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
  xdata = cursor[0]; ydata = cursor[1]
  new_xrange = cur_xrange*scale_factor; new_yrange = cur_yrange*scale_factor
  xdata = min( max( xdata, xlim[0]+new_xrange ), xlim[1]-new_xrange )
  xL = max( xlim[0], xdata - new_xrange ); xR = min( xlim[1], xdata + new_xrange )
  if ylim[1]>ylim[0]:
    ydata = min( max( ydata, ylim[0]+new_yrange ), ylim[1]-new_yrange )
    yL = max( ylim[0], ydata - new_yrange ); yR = min( ylim[1], ydata + new_yrange )
  else:
    ydata = min( max( ydata, ylim[1]-new_yrange ), ylim[0]+new_yrange )
    yR = max( ylim[1], ydata + new_yrange ); yL = min( ylim[0], ydata - new_yrange )
  if xL==cur_xlim[0] and xR==cur_xlim[1] and \
     yL==cur_ylim[0] and yR==cur_ylim[1]: return (None, None), (None, None)
  return (xL, xR), (yL, yR)


def createYZlabels(y, z, ylabel, yunits, zlabel, zunits):
  """
  Checks that y and z labels are appropriate and tries to make some if they are not.
  """
  if y is None:
    if ylabel is None: ylabel='j'
    if yunits is None: yunits=''
  else:
    if ylabel is None: ylabel='Latitude'
    #if yunits is None: yunits=u'\u00B0N'
    if yunits is None: yunits=r'$\degree$N'
  if z is None:
    if zlabel is None: zlabel='k'
    if zunits is None: zunits=''
  else:
    if zlabel is None: zlabel='Elevation'
    if zunits is None: zunits='m'
  return ylabel, yunits, zlabel, zunits

def createTZlabels(t, z, tlabel, tunits, zlabel, zunits):
  """
  Checks that y and z labels are appropriate and tries to make some if they are not.
  """
  if t is None:
    if tlabel is None: tlabel='t'
    if tunits is None: tunits=''
  else:
    if tlabel is None: tlabel='Time'
    if tunits is None: tunits=''
  if z is None:
    if zlabel is None: zlabel='k'
    if zunits is None: zunits=''
  else:
    if zlabel is None: zlabel='Elevation'
    if zunits is None: zunits='m'
  return tlabel, tunits, zlabel, zunits


def yzWeight(y, z):
  """
  Calculates the weights to use when calculating the statistics of a y-z section.

  y(nj+1) is a 1D vector of column edge positions and z(nk+1,nj) is the interface
  elevations of each column. Returns weight(nk,nj).
  """
  dz = z[:-1,:] - z[1:,:]
  return numpy.matlib.repmat(y[1:] - y[:-1], dz.shape[0], 1) * dz


def dunne_rainbow(N=256):
  """
  Spectral/rainbow colormap from John Dunne.
  """
  cdict = {'red': [(0.00, 0.95, 0.95),
                   (0.09, 0.85, 0.85),
                   (0.18, 0.60, 0.60),
                   (0.32, 0.30, 0.30),
                   (0.45, 0.00, 0.00),
                   (0.60, 1.00, 1.00),
                   (0.85, 1.00, 1.00),
                   (1.00, 0.40, 0.00)],

         'green': [(0.00, 0.75, 0.75),
                   (0.09, 0.85, 0.85),
                   (0.18, 0.60, 0.60),
                   (0.32, 0.20, 0.20),
                   (0.45, 0.60, 0.60),
                   (0.60, 1.00, 1.00),
                   (0.73, 0.70, 0.70),
                   (0.85, 0.00, 0.00),
                   (1.00, 0.00, 0.00)],

         'blue':  [(0.00, 1.00, 1.00),
                   (0.32, 1.00, 1.00),
                   (0.45, 0.30, 0.30),
                   (0.60, 0.00, 0.00),
                   (1.00, 0.00, 0.00)]}
  import matplotlib
  cmap = matplotlib.colors.LinearSegmentedColormap('dunneRainbow', cdict, N=N)
  #cmap.set_under([1,.65,.85]); cmap.set_over([.25,0.,0.])
  cmap.set_under([.95*.9,.75*.9,.9]); cmap.set_over([.3,0.,0.])
  #cmap.set_bad('w')
  matplotlib.cm.register_cmap(cmap=cmap)
  return cmap


def dunne_pm(N=256):
  """
  Plus-minus  colormap from John Dunne.
  """
  cdict = {'red':   [(0.00, 0.3, 0.3),
                     (0.05, 0.5, 0.5),
                     (0.20, 0.0, 0.0),
                     (0.30, 0.4, 0.4),
                     (0.40, 0.8, 0.8),
                     (0.50, 1.0, 1.0),
                     (0.95, 0.6, 0.6),
                     (1.00, 0.4, 0.4)],

           'green': [(0.00, 0.0, 0.0),
                     (0.30, 0.5, 0.5),
                     (0.40, 1.0, 1.0),
                     (0.70, 1.0, 1.0),
                     (1.00, 0.0, 0.0)],

           'blue':  [(0.00, 0.3, 0.3),
                     (0.05, 0.5, 0.5),
                     (0.20, 1.0, 1.0),
                     (0.50, 1.0, 1.0),
                     (0.60, 0.7, 0.7),
                     (0.70, 0.0, 0.0),
                     (1.00, 0.0, 0.0)]}
  import matplotlib
  cmap = matplotlib.colors.LinearSegmentedColormap('dunnePM', cdict, N=N)
  cmap.set_under([.1,.0,.1]); cmap.set_over([.2,0.,0.])
  #cmap.set_bad('w')
  matplotlib.cm.register_cmap(cmap=cmap)
  return cmap

def brownblue_cmap():
  # The color map below is from the Light & Bartlein collection
  # which is tested for several different forms of colorblindness
  #
  # Reference:
  # A. Light & P.J. Bartlein, "The End of the Rainbow? Color Schemes for
  # Improved Data Graphics," Eos,Vol. 85, No. 40, 5 October 2004.
  # http://geog.uoregon.edu/datagraphics/EOS/Light-and-Bartlein.pdf

  lb_brownblue_values = numpy.array([
    [144,100, 44],
    [187,120, 54],
    [225,146, 65],
    [248,184,139],
    [244,218,200],
    [255,255,255],   #[241,244,245],
    [207,226,240],
    [160,190,225],
    [109,153,206],
    [70, 99, 174],
    [24, 79, 162]])

  lb_brownblue_values = lb_brownblue_values/255.
  lb_brownblue_values = lb_brownblue_values[::-1,:]

  import matplotlib
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list('brownblue',lb_brownblue_values)
  cmap.set_bad('w')
  matplotlib.cm.register_cmap(cmap=cmap)
  return cmap

def parula_cmap():
  parula_values = numpy.array([
    [0.2081,0.1663,0.5292],
    [0.2116,0.1898,0.5777],
    [0.2123,0.2138,0.6270],
    [0.2081,0.2386,0.6771],
    [0.1959,0.2645,0.7279],
    [0.1707,0.2919,0.7792],
    [0.1253,0.3242,0.8303],
    [0.0591,0.3598,0.8683],
    [0.0117,0.3875,0.8820],
    [0.0060,0.4086,0.8828],
    [0.0165,0.4266,0.8786],
    [0.0329,0.4430,0.8720],
    [0.0498,0.4586,0.8641],
    [0.0629,0.4737,0.8554],
    [0.0723,0.4887,0.8467],
    [0.0779,0.5040,0.8384],
    [0.0793,0.5200,0.8312],
    [0.0749,0.5375,0.8263],
    [0.0641,0.5570,0.8240],
    [0.0488,0.5772,0.8228],
    [0.0343,0.5966,0.8199],
    [0.0265,0.6137,0.8135],
    [0.0239,0.6287,0.8038],
    [0.0231,0.6418,0.7913],
    [0.0228,0.6535,0.7768],
    [0.0267,0.6642,0.7607],
    [0.0384,0.6743,0.7436],
    [0.0590,0.6838,0.7254],
    [0.0843,0.6928,0.7062],
    [0.1133,0.7015,0.6859],
    [0.1453,0.7098,0.6646],
    [0.1801,0.7177,0.6424],
    [0.2178,0.7250,0.6193],
    [0.2586,0.7317,0.5954],
    [0.3022,0.7376,0.5712],
    [0.3482,0.7424,0.5473],
    [0.3953,0.7459,0.5244],
    [0.4420,0.7481,0.5033],
    [0.4871,0.7491,0.4840],
    [0.5300,0.7491,0.4661],
    [0.5709,0.7485,0.4494],
    [0.6099,0.7473,0.4337],
    [0.6473,0.7456,0.4188],
    [0.6834,0.7435,0.4044],
    [0.7184,0.7411,0.3905],
    [0.7525,0.7384,0.3768],
    [0.7858,0.7356,0.3633],
    [0.8185,0.7327,0.3498],
    [0.8507,0.7299,0.3360],
    [0.8824,0.7274,0.3217],
    [0.9139,0.7258,0.3063],
    [0.9450,0.7261,0.2886],
    [0.9739,0.7314,0.2666],
    [0.9938,0.7455,0.2403],
    [0.9990,0.7653,0.2164],
    [0.9955,0.7861,0.1967],
    [0.9880,0.8066,0.1794],
    [0.9789,0.8271,0.1633],
    [0.9697,0.8481,0.1475],
    [0.9626,0.8705,0.1309],
    [0.9589,0.8949,0.1132],
    [0.9598,0.9218,0.0948],
    [0.9661,0.9514,0.0755],
    [0.9763,0.9831,0.0538]])

  import matplotlib
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list('parula',parula_values)
  cmap.set_bad('w')
  matplotlib.cm.register_cmap(cmap=cmap)
  return cmap

def plotBasemapPanel(field, sector, xCoord, yCoord, lonRange, latRange, cmap, norm, interactive, extend):
  if sector == 'arctic':  m = Basemap(projection='npstere',boundinglat=60,lon_0=-120,resolution='l')
  elif sector == 'shACC': m = Basemap(projection='spstere',boundinglat=-45,lon_0=-120,resolution='l')
  else:  m = Basemap(projection='mill',lon_0=-120.,resolution='l',llcrnrlon=lonRange[0], \
             llcrnrlat=latRange[0], urcrnrlon=lonRange[1],urcrnrlat=latRange[1])
  m.drawmapboundary(fill_color='0.85')
  im0 = m.pcolormesh(numpy.minimum(xCoord,60.),yCoord,(field),shading='flat',cmap=cmap,norm=norm,latlon=True)
  m.drawcoastlines()
  if interactive: addStatusBar(xCoord, yCoord, field)
  if extend is None: extend = extend
  cb1 = m.colorbar(pad=0.15, extend=extend)
  if sector == 'tropPac': drawNinoBoxes(m)


def drawNinoBoxes(m,region='all'):
  '''
  Function to draw ENSO region boxes on a basemap instance
  '''
  if region == 'nino4' or region == 'all':
    polyLon = [-200., -200., -150., -150., -200.]
    polyLat = [-5., 5., 5., -5., -5.]
    polyX, polyY = m(polyLon,polyLat)
    m.plot(polyX, polyY, marker=None,color='k',linewidth=2.0)
  if region == 'nino3' or region == 'all':
    polyLon = [-150., -150., -90., -90., -150.]
    polyLat = [-5., 5., 5., -5., -5.]
    polyX, polyY = m(polyLon,polyLat)
    m.plot(polyX, polyY, marker=None,color='k',linewidth=2.0)
  if region == 'nino34' or region == 'all':
    polyLon = [-170., -170., -120., -120., -170.]
    polyLat = [-5., 5., 5., -5., -5.]
    polyX, polyY = m(polyLon,polyLat)
    m.plot(polyX, polyY, marker=None,color='r',linestyle='dashed',linewidth=2.0)
  if region == 'nino12' or region == 'all':
    polyLon = [-90., -90., -80., -80., -90.]
    polyLat = [-10., 0., 0., -10., -10.]
    polyX, polyY = m(polyLon,polyLat)
    m.plot(polyX, polyY, marker=None,color='k',linewidth=2.0)


def regionalMasking(field,yCoord,xCoord,latRange,lonRange):
    maskedField = field.copy()
    maskedField = numpy.ma.masked_where(numpy.less(yCoord[0:-1,0:-1],latRange[0]), maskedField)
    maskedField = numpy.ma.masked_where(numpy.greater(yCoord[1::,1::],latRange[1]),maskedField)
    maskedField = numpy.ma.masked_where(numpy.less(xCoord[0:-1,0:-1],lonRange[0]), maskedField)
    maskedField = numpy.ma.masked_where(numpy.greater(xCoord[1::,1::],lonRange[1]),maskedField)
    return maskedField

def cmoceanRegisterColormaps():
  import matplotlib
  matplotlib.cm.register_cmap(name='algae',cmap=cmocean.cm.algae)
  matplotlib.cm.register_cmap(name='thermal',cmap=cmocean.cm.thermal)
  matplotlib.cm.register_cmap(name='haline',cmap=cmocean.cm.haline)
  matplotlib.cm.register_cmap(name='solar',cmap=cmocean.cm.solar)
  matplotlib.cm.register_cmap(name='ice',cmap=cmocean.cm.ice)
  matplotlib.cm.register_cmap(name='gray',cmap=cmocean.cm.gray)
  matplotlib.cm.register_cmap(name='oxy',cmap=cmocean.cm.oxy)
  matplotlib.cm.register_cmap(name='deep',cmap=cmocean.cm.deep)
  matplotlib.cm.register_cmap(name='dense',cmap=cmocean.cm.dense)
  matplotlib.cm.register_cmap(name='algae',cmap=cmocean.cm.algae)
  matplotlib.cm.register_cmap(name='matter',cmap=cmocean.cm.matter)
  matplotlib.cm.register_cmap(name='turbid',cmap=cmocean.cm.turbid)
  matplotlib.cm.register_cmap(name='speed',cmap=cmocean.cm.speed)
  matplotlib.cm.register_cmap(name='amp',cmap=cmocean.cm.amp)
  matplotlib.cm.register_cmap(name='tempo',cmap=cmocean.cm.tempo)
  matplotlib.cm.register_cmap(name='phase',cmap=cmocean.cm.phase)
  matplotlib.cm.register_cmap(name='balance',cmap=cmocean.cm.balance)
  matplotlib.cm.register_cmap(name='delta',cmap=cmocean.cm.delta)
  matplotlib.cm.register_cmap(name='curl',cmap=cmocean.cm.curl)


def sectorRanges(sector=None):
  # Should add definitions for tropInd, tropAtl, sAtl, nPac, allPac, allInd, and allAtlantic
  if sector   == 'nAtl':    lonRange=(-100.,40.); latRange=(-15.,80.); hspace=0.25; titleOffset=1.14
  elif sector == 'gomex':   lonRange=(-100.,-50.); latRange=(5.,35.); hspace=0.25; titleOffset=1.14
  elif sector == 'tropPac': lonRange=(-270.,-75.); latRange=(-30.,30.); hspace=-0.2; titleOffset=1.17
  elif sector == 'arctic':  lonRange=(-300,60); latRange=(60.,90.); hspace=0.25; titleOffset=1.14
  elif sector == 'shACC':   lonRange=(-300,60); latRange=(-90,-45.); hspace=0.25; titleOffset=1.14
  else: lonRange=None; latRange=None; hspace=0.25; titleOffset=1.14
  return lonRange, latRange, hspace, titleOffset

# Load new named colormaps
c = dunne_rainbow()
c = dunne_pm()
c = brownblue_cmap()
c = parula_cmap()

# Register cmocean colormaps
#if 'cmocean' in modules.keys():
#  cmoceanRegisterColormaps()

# Test
if __name__ == '__main__':
  import nccf
  file = 'baseline/19000101.ocean_static.nc'
  D,(y,x),_ = nccf.readVar(file,'depth_ocean')
  y,_,_ = nccf.readVar(file,'geolat')
  x,_,_ = nccf.readVar(file,'geolon')
  area,_,_ = nccf.readVar(file,'area_t')
  xyplot(D, x, y, title='Depth', ignore=0, suptitle='Testing', area=area, clim=[0, 5500], nbins=12, debug=True, interactive=True, show=True)#, save='fig_test.png')
  xycompare(D, .9*D, x, y, title1='Depth', ignore=0, suptitle='Testing', area=area, nbins=12)#, save='fig_test2.png')
  annual = 'baseline/19000101.ocean_annual.nc'
  monthly = 'baseline/19000101.ocean_month.nc'
  e,(t,z,y,x),_ = nccf.readVar(annual,'e',0,None,None,1100)
  temp,(t,z,y,x),_ = nccf.readVar(monthly,'temp',0,None,None,1100)
  temp2,(t,z,y,x),_ = nccf.readVar(monthly,'temp',11,None,None,1100)
  yzplot(temp, y, e)
  yzcompare(temp, temp2, y, e, interactive=True)
  yzcompare(temp, temp2, y, e, npanels=2)
  yzcompare(temp, temp2, y, e, npanels=1)
  plt.show()
