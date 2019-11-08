#!/usr/bin/env python

# <h1 align="center">MOM6 diagnostic boundary fluxes of scalars and their global budgets</h1>
#
# Results from this notebook:
# 1. Maps of surface boundary fluxes of water, heat, and salt crossing into the liquid seawater in MOM6;
# 2. Computation of self-consistency checks, including global heat, salt, and mass budgets to verify that the model is conserving scalars over the global domain.
#
# Caveats regarding this notebook:
# 1. This notebook is written for the MOM6-examples/ocean_only/global_ALE/z and
#    MOM6-examples/ocean_only/global_ALE/layer test cases.
#    It is nearly the same as the notebook for MOM6-examples/ice_ocean_SIS/GOLD_SIS.
# 2. It only considers tendencies over a single time step.
#
# Hopes for the use of this notebook:
# 1. To provide a starting point to document boundary fluxes of scalar fields;
# 2. To teach MOM6 users about the boundary fluxes, their patterns, units, and sign conventions;
# 3. To perform self-consistency checks to ensure the model is conserving scalar fields;
# 4. To illustrate a self-contained iPython notebook of use for MOM6 analysis.
#
# This iPython notebook was originally developed at NOAA/GFDL, and it is provided freely to the MOM6 community. GFDL scientists developing MOM6 make extensive use of Python for diagnostics. We solicit modifications/fixes that are useful to the MOM6 community.

import matplotlib.pyplot as plt
import netCDF4
import numpy
import sys
try: import argparse
except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')

def run():
  parser = argparse.ArgumentParser(description='''Script for diagnosing boundary fluxes of scalars and their global budgets''')
  parser.add_argument('dir', type=str, help='''Directory where MOM netCDF files are located.''')
  parser.add_argument('--plot', help='''If defined, it will plot the fields.''', action='store_true')
  parser.add_argument('-n','--name', type=str, default='', help='''Case name (default is '')''')
  parser.add_argument('-f','--forcing', type=str, default='mom6.frc_0001.nc', help='''Name of the forcing file (default is ocean.mom6.frc_0001.nc)''')
  parser.add_argument('-s','--surface', type=str, default='mom6.sfc_0001.nc', help='''Name of the surface file (default is ocean.mom6.sfc_0001.nc)''')
  parser.add_argument('-g','--geometry', type=str, default='mom6.static.nc', help='''Name of the file with the geometric variables (default is ocean.mom6.static.nc)''')
  cmdLineArgs = parser.parse_args()
  main(cmdLineArgs)

def main(cmdLineArgs):
  plot = cmdLineArgs.plot
  path    = cmdLineArgs.dir+'/'+cmdLineArgs.name+'.'
  static  = netCDF4.Dataset(path+cmdLineArgs.geometry)
  forcing = netCDF4.Dataset(path+cmdLineArgs.forcing)
  surface = netCDF4.Dataset(path+cmdLineArgs.surface)

  # define variable names and parameters
  tvar = 'time'
  cp   = 3992.0
  rho0 = 1035.0
  n    = 2
  print('Variables saved:')
  for v in forcing.variables: print (v),
  for v in surface.variables: print (v),
  for v in static.variables: print (v),
  print('\n \n')

  # This section fills the fields used in this analysis.
  #--------------------------------------------------------------
  # geometric factors
  lon  = static.variables['geolon'][:]
  lat  = static.variables['geolat'][:]
  wet  = static.variables['wet'][:]
  global area
  area = static.variables['area_t'][:]*wet
  #--------------------------------------------------------------
  # time in days, convert to seconds
  time = surface.variables[tvar][:]*86400.0000000000000
  #--------------------------------------------------------------
  # sea surface temperature
  sst = surface.variables['SST'][n]
  #--------------------------------------------------------------
  # \int \rho \, dz \, (1,Theta,S), with \rho = \rho_{0} for Bousssinesq.
  mass_wt = surface.variables['mass_wt']
  tomint  = surface.variables['temp_int']
  somint  = surface.variables['salt_int']
  #tomint  = surface.variables['tomint']
  #somint  = surface.variables['somint']
  #--------------------------------------------------------------
  # mass flux of water crossing ocean surface [kg/(m^2 s)]
  # positive values indicate mass entering ocean;
  # negative values indicate mass leaving ocean.
  # net mass flux entering ocean
  net_massin = forcing.variables['net_massin'][n]
  # net mass flux leaving ocean
  net_massout = forcing.variables['net_massout'][n]
  # evaporation (negative) and condensation (positive)
  evap = forcing.variables['evap'][n]
  # seaice_melt, fresh water from melting/forming sea ice
  seaice_melt = forcing.variables['seaice_melt'][n]
  # liquid runoff entering ocean (non-negative)
  lrunoff = forcing.variables['lrunoff'][n]
  # frozen runoff entering ocean (non-negative)
  frunoff = forcing.variables['frunoff'][n]
  # liquid precipitation entering ocean.
  # note: includes exchanges with sea-ice, with
  #       melt adding mass to ocean; formation removing mass.
  lprec = forcing.variables['lprec'][n]
  # frozen precipitation entering ocean.
  fprec = forcing.variables['fprec'][n]
  # virtual precipitation arising from conversion of salt restoring to water flux
  vprec = forcing.variables['vprec'][n]
  # net mass flux crossing surface (including exchange with sea-ice)
  PRCmE = forcing.variables['PRCmE'][n]

  #--------------------------------------------------------------
  # heat flux crossing ocean surface and bottom [Watt/m^2]
  # positive values indicate heat entering ocean;
  # negative values indicate heat leaving ocean.
  # net heat crossing ocean surface due to all processes, except restoring
  net_heat_surface = forcing.variables['net_heat_surface'][n]
  # net heat passed through coupler from shortwave, longwave, latent, sensible.
  # note: latent includes heat to vaporize liquid and heat to melt ice/snow.
  # note: sensible includes air-sea and ice-sea sensible heat fluxes.
  net_heat_coupler = forcing.variables['net_heat_coupler'][n]
  # sum of longwave + latent + sensible
  LwLatSens = forcing.variables['LwLatSens'][n]
  # net shortwave passing through ocean surface
  SW = forcing.variables['SW'][n]
  # heating of liquid seawater due to formation of frazil sea ice
  frazil = forcing.variables['frazil'][n]
  # heat from melting/forming sea ice
  seaice_melt_heat = forcing.variables['seaice_melt_heat'][n]
  # net heat content associated with transfer of mass across ocean surface,
  # computed relative to 0C. Both diagnostics should be the same, though
  # they are computed differently in MOM6.
  heat_pme = forcing.variables['Heat_PmE'][n]
  heat_content_surfwater = forcing.variables['heat_content_surfwater'][n]
  # heat content associated with water mass leaving ocean
  heat_content_massout = forcing.variables['heat_content_massout'][n]
  # heat content associated with water mass entering ocean
  heat_content_massin = forcing.variables['heat_content_massin'][n]
  # heat content associated with liquid precipitation
  heat_content_lprec = forcing.variables['heat_content_lprec'][n]
  # heat content associated with meltw
  heat_content_icemelt = forcing.variables['heat_content_icemelt'][n]
  # heat content associated with frozen precipitation
  heat_content_fprec = forcing.variables['heat_content_fprec'][n]
  # heat content associated with virtual precipitation
  heat_content_vprec = forcing.variables['heat_content_vprec'][n]
  # heat content associated with liquid runoff
  heat_content_lrunoff = forcing.variables['heat_content_lrunoff'][n]
  # heat content associated with frozen runoff
  heat_content_frunoff = forcing.variables['heat_content_frunoff'][n]
  # heat content associated with liquid condensation
  heat_content_cond = forcing.variables['heat_content_cond'][n]
  #--------------------------------------------------------------
  # salt flux crossing ocean surface and bottom [kg/(m^2 s)]
  # positive values indicate salt entering ocean;
  # negative values indicate salt leaving ocean.

  # salt flux arising from ocean-ice interactions.
  # this term is zero for this test case, as there is no icea model.
  #salt_flux = forcing.variables['salt_flux'][n]
  salt_flux  = forcing.variables['salt_flux_in'][n]
  # salt flux associated with surface restoring.
  # salt_flux has contribution from sea ice + restoring, so we need to remove salt_flux (salt_flux_in)
  salt_restore = forcing.variables['salt_flux'][n] - salt_flux
  # SSH analysis
  ssh = surface.variables['SSH'][:]
  tm, jm, im = ssh.shape
  ssh_tend = numpy.zeros(tm)
  for t in range(tm):
    ssh_tend[t] = (ssh[t,:]*area).sum()/area.sum()
    #print('SSH tendency (m):',ssh_tend[t])

  # <h1 align="center">Mass fluxes and global seawater mass budget</h1>

  # <h2 align="center">Global seawater mass budget consistency check</h2>
  #
  # We compute the change in seawater mass over a given time period.  Two different methods are used, and the two methods should agree at the level of truncation error.  Note that "truncation error" precision is somewhat larger using offline diagnostics relative to online calculations, particularly if single precision output is saved rather than double precision.
  #
  # The net mass per time of water (units of kg/s) entering through the ocean boundaries is given by the area integral
  # $$\begin{equation*}
  # \mbox{boundary water mass entering liquid seawater} = \int Q_{W} \, dA,
  # \end{equation*}$$
  # where the net water flux (units of $\mbox{kg}~\mbox{m}^{-2}~\mbox{s}^{-1}$) is given by
  # $$\begin{align*}
  #  Q_{W} &= {\tt PRCmE}
  # \end{align*}$$
  # A nonzero surface mass flux is associated with liquid and solid precipitation and runoff; evaporation and condensation; sea ice melt/formation; and surface restoring.
  #
  # The time change of liquid seawater mass is computed according to
  # $$\begin{equation*}
  # \mbox{seawater mass change} =
    # \frac{1}{\tau_{n+1} - \tau_{n} } \int dA \left(\int (\rho_{n+1} - \rho_{n}) \, \mathrm{d}z \right)
  # \end{equation*}$$
  # where $\tau_{n+1} - \tau_{n}$ is the time increment in seconds.  Note that we make use of the MOM6 diagnostic for depth integrated density
  # $$\begin{equation*}
  #  {\tt mass\_wt} =  \int \rho \, \mathrm{d}z.
  # \end{equation*}$$
  # For a Boussinesq fluid, the in-situ $\rho$ factor is set to $\rho_{0}$, in which case the diagnostic field {\tt mass\_wt} measures the thickness of a fluid column, multiplied by $\rho_{0}$.  For self-consistency, we should have the following equality holding to within truncation error
  # $$\begin{equation*}
  # \boxed{
  #  \mbox{boundary water mass entering liquid seawater} = \mbox{seawater mass change}.
  # }
  # \end{equation*}$$

  n0       = n-1
  dmass_wt = mass_wt[n] - mass_wt[n0]
  dt = time[n] - time[n0]
  lhs = area * dmass_wt / dt
  rhs = area * ( PRCmE )
  print ('Total seawater mass at time step n  [kg seawater]  =',(mass_wt[n]*area).sum())
  print ('Total seawater mass at time step n0 [kg seawater]  =',(mass_wt[n0]*area).sum())
  print ('Total seawater mass content change [kg seawater]   =',dt*lhs.sum())
  print ('Net water mass through boundaries [kg seawater]    =',dt*rhs.sum())
  print ('Residual [kg seawater]                             =',dt*lhs.sum() - dt*rhs.sum())
  print ('Non-dimensional residual                           =',(  lhs.sum() - rhs.sum() )/lhs.sum())
  print ('Non-dimensional residual                           =',(  lhs.sum() - rhs.sum() )/(mass_wt[n]*area).sum())
  print('\n')

  var = ( lhs.sum() - rhs.sum())/(mass_wt[n]*area).sum()
  if var >= numpy.finfo(float).eps:
    print('Mass is not conserved!')
    sys.exit()

  if plot:
    #Surface mass fluxes I: combined fields
    plt.figure(figsize=(16,14))
    newSP(2,2);
    field     = 86400.0*PRCmE
    make_plot(lon,lat,field, '$PRCmE$ [$kg/m^2/day$]',cmin=-20,cmax=20)
    nextSP()
    field     = 86400.0*net_massin
    make_plot(lon,lat,field, 'net_massin [$kg/m^2/day$]',cmin=-20,cmax=20)
    nextSP()
    field     = 86400.0*net_massout
    make_plot(lon,lat,field, 'net_massout [$kg/m^2/day$]',cmin=-20,cmax=20,xlabel=True)
    nextSP()
    field     = 86400.0*(PRCmE - net_massout - net_massin)
    make_plot(lon,lat,field, '$PRCmE - M_{in} - M_{out}$ [$kg/m^2/day$]',cmin=-1e-13,cmax=1e-13,xlabel=True)
    plt.show()

    # Surface mass fluxes II: component fields
    plt.figure(figsize=(16,12))
    newSP(3,2);
    field     = 86400.0*seaice_melt
    make_plot(lon,lat,field, 'seaice_melt [$kg/m^2/day$]',cmin=-10,cmax=10)
    nextSP()
    field     = 86400.0*lrunoff
    make_plot(lon,lat,field, 'lrunoff [$kg/m^2/day$]',cmin=-10,cmax=10)
    #nextSP()
    #field     = 86400.0*frunoff
    nextSP()
    # lprec contains a contribution from sea ice melt/formation.
    field     = 86400.0*lprec
    make_plot(lon,lat,field, 'lprec [$kg/m^2/day$]',cmin=-10,cmax=10)
    nextSP()
    field     = 86400.0*fprec
    make_plot(lon,lat,field, 'fprec [$kg/m^2/day$]',cmin=-1,cmax=1)
    nextSP()
    # evaporation and condensation
    field     = 86400.0*evap
    make_plot(lon,lat,field, 'evap [$kg/m^2/day$]',cmin=-10,cmax=10,xlabel=True)
    nextSP()
    field     = 86400.0*vprec
    make_plot(lon,lat,field, 'vprec [$kg/m^2/day$]',cmin=-10,cmax=10,xlabel=True)
    plt.show()

    # Surface mass flux self-consistency check
    plt.figure(figsize=(16,12))
    newSP(1,1);
    field     = 86400.0*(PRCmE -lprec -fprec -lrunoff -frunoff -vprec -evap -seaice_melt)
    make_plot(lon,lat,field, 'PRCmE -lprec -fprec -lrunoff -frunoff -vprec -evap -seaice_melt [$kg/m^2/day$]',cmin=-1e-13,cmax=1e-13)
    plt.show()
  # <h1 align="center">Heat fluxes and global ocean heat budget</h1>
  # <h2 align="center">Global heat budget consistency check</h2>
  # We compute the change in seawater heat content over a given time period.  Two different methods are used, and the two methods should agree at the level of truncation error.  If larger differences exist, then there is a bug.
  #
  # The net heat per time (units of Watts) entering through the ocean boundaries is given by the area integral
  # $$\begin{equation*}
  # \mbox{boundary heating of liquid seawater} = \int Q \, dA,
  # \end{equation*}$$
  # where the net heat flux (units of $\mbox{W}~\mbox{m}^{-2}~\mbox{s}^{-1}$) is given by
  # $$\begin{align*}
  #  Q &= {\tt (net\_heat\_coupler + heat\_pme + frazil) + internal\_heat + heat\_restore} \\
  #    &= {\tt net\_heat\_surface + internal\_heat + heat\_restore}
  # \end{align*}$$
  # The time change of liquid seawater heat is computed according to
  # $$\begin{equation*}
  # \mbox{seawater heat content change} =
  # \frac{C_p }{\tau_{n+1} - \tau_{n} } \int dA \left(\rho_0 \int (\Theta_{n+1} - \Theta_{n}) \, \mathrm{d}z \right)
  # \end{equation*}$$
  #  where $\tau_{n+1} - \tau_{n}$ is the time increment in seconds.  Note that we make use of the MOM6 diagnostic for depth integrated potential/conservative temperature
  # $$\begin{equation*}
  #  {\tt tomint} = \rho_0 \int \Theta \, \mathrm{d}z,
  # \end{equation*}$$
  #  where the Boussinesq reference density, $\rho_{0}$, is used since this test case makes the Boussinesq approximation. For self-consistency, we should have the following equality holding to within truncation error
  # $$\begin{equation*}
  # \boxed{
  #  \mbox{boundary heating of liquid seawater} = \mbox{seawater heat content change}.
  # }
  # \end{equation*}$$

  n0      = n-1
  dtomint = tomint[n] - tomint[n0]
  dt      = time[n] - time[n0]
  lhs     = cp * area * dtomint / dt
  rhs     = area * ( net_heat_coupler + heat_pme + frazil)

  print ('Total seawater heat at time step n  [Joules]  =',cp * (area * tomint[n]).sum())
  print ('Total seawater heat at time step n0 [Joules]  =',cp * (area* tomint[n0]).sum())
  print ('Total seawater heat content change [Joules]   =',dt*lhs.sum())
  print ('Net heat through boundaries [Joules]          =',dt*rhs.sum())
  print ('Residual [Joules]                             =',dt*lhs.sum() - dt*rhs.sum())
  print ('Non-dimensional residual                      =',( lhs.sum() - rhs.sum() )/lhs.sum())
  print ('Non-dimensional residual                      =',( lhs.sum() - rhs.sum() )/(cp * (area * tomint[n]).sum()))
  print('\n')
  var = ( lhs.sum() - rhs.sum() )/(cp * (area * tomint[n]).sum())
  if var >= numpy.finfo(float).eps:
    print('Heat is not conserved!')
    sys.exit()

  if plot:
    # Basic components to surface heat flux
    # self-consistency check
    # net_heat_surface = heat_pme + frazil + net_heat_coupler + heat_restore
    plt.figure(figsize=(16,12))
    newSP(3,2);
    field     = net_heat_surface
    make_plot(lon,lat,field, 'net_heat_surface[$W/m^2$]')
    nextSP()
    field     = net_heat_coupler
    make_plot(lon,lat,field, 'net_heat_coupler[$W/m^2$]')
    nextSP()
    field     = frazil
    make_plot(lon,lat,field, 'frazil[$W/m^2$]',cmin=-20, cmax=20)
    nextSP()
    field     = heat_pme
    make_plot(lon,lat,field, 'heat_pme[$W/m^2$]',cmin=-20,cmax=20)
    nextSP()
    field = net_heat_surface-net_heat_coupler-frazil-heat_pme
    make_plot(lon,lat,field, 'Residual(error)[$W/m^2$]',cmin=0.0,cmax=0.0,xlabel=True)
    plt.show()

    # Heat fluxes crossing ocean surface via the coupler
    # Heat fluxes crossing ocean surface via the coupler
    # net_heat_coupler =  LwLatSens + SW + seaice_melt_heat
    # LwLatSens = LW + Latent + Sensible
    plt.figure(figsize=(16,12))
    newSP(3,2);
    field     = net_heat_coupler
    make_plot(lon,lat,field, 'net_heat_coupler[$W/m^2$]',cmin=-200,cmax=200)
    nextSP()
    field     = LwLatSens
    make_plot(lon,lat,field, 'LwLatSens [$W/m^2$]',cmin=-200,cmax=200)
    nextSP()
    field     = SW
    make_plot(lon,lat,field, 'SW [$W/m^2$]',cmin=-200,cmax=200)
    nextSP()
    field     = seaice_melt_heat
    make_plot(lon,lat,field, 'seaice_melt_heat [$W/m^2$]',cmin=-200,cmax=20, xlabel=True)
    nextSP()
    field     = net_heat_coupler - SW - LwLatSens - seaice_melt_heat
    make_plot(lon,lat,field, 'Residual(error) [$W/m^2$]',cmin=0.0,cmax=0.0, xlabel=True)
    plt.show()

    # Relation between heat_PmE, heat_content_massin, and heat_content_massout
    # Alternative means to compute to heat_PmE via
    # heat_content_massin and heat_content_massout
    # heat_PmE = heat_content_massin + heat_content_massout
    plt.figure(figsize=(18,12))
    newSP(3,2);
    field     = heat_content_massout
    make_plot(lon,lat,field, 'heat_content_massout [$W/m^2$]',cmin=-20,cmax=0)
    nextSP()
    field     = heat_content_massin
    make_plot(lon,lat,field, 'heat_content_massin [$W/m^2$]',cmin=0,cmax=20)
    nextSP()
    field     = heat_content_massin + heat_content_massout
    make_plot(lon,lat,field, 'heat_content_massin + heat_content_massout [$W/m^2$]',cmin=-20,cmax=20)
    nextSP()
    field     = heat_pme
    make_plot(lon,lat,field, 'heat_pme [$W/m^2$]',cmin=-20,cmax=20,xlabel=True)
    #nextSP()
    #field     = seaice_melt_heat
    #make_plot(lon,lat,field, 'Melth [$W/m^2$]',cmin=-20.0,cmax=20.0, xlabel=True)
    nextSP()
    field     = heat_content_massout + heat_content_massin - heat_pme
    make_plot(lon,lat,field, 'heat_massin + heat_massout - heat_pme [$W/m^2$]',cmin=0.0,cmax=0.0, xlabel=True)
    plt.show()

    # Components of heat content from surface mass fluxes
    # Components of heat content of surface mass fluxes
    # heat_PmE = heat_content_lprec + heat_content_fprec + heat_content_vprec
    #          + heat_content_lrunoff + heat_content_melth + heat_content_cond + heat_content_massout
    plt.figure(figsize=(16,12))
    newSP(3,2);
    field     = heat_content_lprec
    make_plot(lon,lat,field, 'heat_content_lprec [$W/m^2$]',cmin=-20.0,cmax=20.0)
    nextSP()
    field     = heat_content_lrunoff
    make_plot(lon,lat,field, 'heat_content_lrunoff [$W/m^2$]',cmin=-20.0,cmax=20.0)
    nextSP()
    field     = heat_content_icemelt
    make_plot(lon,lat,field, 'heat_content_icemelt [$W/m^2$]',cmin=-20.0,cmax=20.0)
    nextSP()
    field     = heat_content_cond
    make_plot(lon,lat,field, 'heat_content_cond [$W/m^2$]',cmin=-1.0,cmax=1.0)
    nextSP()
    field     = heat_content_fprec
    make_plot(lon,lat,field, 'heat_content_fprec [$W/m^2$]',cmin=-1.0,cmax=1.0)
    nextSP()
    field     = heat_content_vprec
    make_plot(lon,lat,field, 'heat_content_vprec [$W/m^2$]',cmin=-20.0,cmax=20.0)
    plt.show()

    # Self-consistency of diagnosed heat content from mass entering ocean
    plt.figure(figsize=(16,12))
    newSP(2,2);
    heat_content_sum = ( heat_content_lprec + heat_content_fprec + heat_content_vprec +
			 heat_content_lrunoff + heat_content_cond + heat_content_icemelt)
    field     = heat_content_massin
    make_plot(lon,lat,field, 'heat_content_massin [$W/m^2$]',cmin=-20.0,cmax=20.0)
    nextSP()
    field     = heat_content_sum
    make_plot(lon,lat,field, 'heat_content_sum [$W/m^2$]',cmin=-20.0,cmax=20.0)
    nextSP()
    field     = heat_content_massin - heat_content_sum
    make_plot(lon,lat,field, 'heat_content_massin - heat_content_sum [$W/m^2$]',cmin=0.0,cmax=0.0)
    plt.show()

    # Self-consistency between heat_pme and heat_content_surfwater
    comp_sum = ( heat_content_lprec + heat_content_fprec + heat_content_vprec + heat_content_lrunoff
	       + heat_content_cond + heat_content_massout + heat_content_icemelt)
    plt.figure(figsize=(16,12))
    newSP(3,2);
    field     = heat_content_surfwater
    make_plot(lon,lat,field, 'heat_content_surfwater [$W/m^2$]',cmin=-20.0,cmax=20.0)
    nextSP()
    field     = comp_sum
    make_plot(lon,lat,field, 'Sum of heat_content components [$W/m^2$]',cmin=-20.0,cmax=20.0)
    nextSP()
    field     = comp_sum - heat_content_surfwater
    make_plot(lon,lat,field, 'Sum of heat_content components - heat_content_surfwater [$W/m^2$]',cmin=0.0,cmax=0.0)
    nextSP()
    field     = heat_pme
    make_plot(lon,lat,field, 'heat_pme [$W/m^2$]',cmin=-20.0,cmax=20.0)
    nextSP()
    field     = heat_pme - heat_content_surfwater
    make_plot(lon,lat,field, 'heat_pme - heat_content_surfwater [$W/m^2$]',cmin=0.0,cmax=0.0)
    plt.show()

    # Map effective temperatures
    # The following "effective" temperatures differ generally from the SST due to the means
    # by which the water is exchanged across the ocean surface boundary. In particular, there
    # are some occasions whereby layers deeper than k=1 need to be sampled in order to exchange
    # water with the atmosphere and/or sea ice.  These maps provide us with a sense for where
    # such occurrances take place.
    TinEff  = heat_content_massin/(net_massin*cp)
    ToutEff = heat_content_massout/(net_massout*cp)
    TnetEff = heat_pme/(PRCmE*cp)
    plt.figure(figsize=(16,12))
    newSP(3,2);
    field     = TinEff
    make_plot(lon,lat,field, '$\Theta_{in} [{\degree}C$]',cmin=-2,cmax=30)
    nextSP()
    field     = TinEff - sst
    make_plot(lon,lat,field, '$\Theta_{in} - SST [{\degree}C$]',cmin=0.0,cmax=0.0)
    nextSP()
    field     = ToutEff
    make_plot(lon,lat,field, '$\Theta_{out} [{\degree}C$]',cmin=-2,cmax=30)
    nextSP()
    field     = ToutEff - sst
    make_plot(lon,lat,field, '$\Theta_{out} - SST [{\degree}C$]',cmin=0.0,cmax=0.0)
    nextSP()
    field     = TnetEff
    make_plot(lon,lat,field, '$\Theta_{net} [{\degree}C$]',cmin=-2,cmax=30, xlabel=True)
    nextSP()
    field     = TnetEff - sst
    make_plot(lon,lat,field, '$\Theta_{net} - SST [{\degree}C$]',cmin=0.0,cmax=0.0, xlabel=True)
    plt.show()

  # Salt fluxes and global ocean salt budget
  # Global salt budget consistency check
  #
  # We compute the change in seawater salt content over a given time period.  Two different methods are used, and the two methods should agree at the level of truncation error.  If larger differences exist, then there is a bug.
  #
  # The net salt per time (units of kg/s) entering through the ocean boundaries is given by the area integral
  # $$\begin{equation*}
  # \mbox{boundary salt entering liquid seawater} = \int Q_{S} \, dA,
  # \end{equation*}$$
  # where the net salt flux (units of $\mbox{kg}~\mbox{m}^{-2}~\mbox{s}^{-1}$) is given by
  # $$\begin{align*}
  #  Q_{S} &= {\tt salt\_flux} + {\tt salt\_restore}.
  # \end{align*}$$
  # A nonzero salt flux is associated with exchanges between liquid seawater and solid sea ice.  It also arises from simulations using a restoring boundary flux associated with damping to observed sea surface salinity.  Finally, there can be a salt flux when using sponges to damp the ocean interior back to an observed value.
  #
  # The time change of liquid seawater salt content is computed according to
  # $$\begin{equation*}
  # \mbox{seawater salt content change} =
  # \frac{1}{\tau_{n+1} - \tau_{n} } \int dA \left(\rho_0 \int (S_{n+1} - S_{n}) \, \mathrm{d}z \right)
  # \end{equation*}$$
  # where $\tau_{n+1} - \tau_{n}$ is the time increment in seconds.  Note that we make use of the MOM6 diagnostic for depth integrated salinity
  # $$\begin{equation*}
  #  {\tt somint} = \rho_0 \int S \, \mathrm{d}z,
  # \end{equation*}$$
  # where the Boussinesq reference density, $\rho_{0}$, is used since this test case makes the Boussinesq approximation. For self-consistency, we should have the following equality holding to within truncation error
  # $$\begin{equation*}
  # \boxed{
  #  \mbox{boundary salt entering liquid seawater} = \mbox{seawater salt content change}.
  # }
  # \end{equation*}$$

  n0      = n-1
  dsomint = somint[n] - somint[n0]
  time    = surface.variables[tvar][:]*86400.
  dt      = time[n] - time[n0]
  lhs     = 1.e-3 * area * dsomint / dt
  rhs     = area * ( salt_flux + salt_restore )

  print('Salt budget:')
  print ('Total seawater salt at time step n  [kg salt] =',(area*somint[n]).sum())
  print ('Total seawater salt at time step n0 [kg salt] =',(area*somint[n0]).sum())
  print ('Total seawater salt content change [kg salt]  =',dt*lhs.sum())
  print ('Net salt through boundaries [kg salt]         =',dt*rhs.sum())
  print ('Residual [kg salt]                            =',dt*lhs.sum() - dt*rhs.sum())
  print ('Non-dimensional residual (based on dsomint)    =',( lhs.sum() - rhs.sum() )/lhs.sum())
  print ('Non-dimensional residual (based on somint[n])    =',( lhs.sum() - rhs.sum() )/((area * somint[n]).sum()))
  print('\n')
  var = ( lhs.sum() - rhs.sum() )/((area * somint[n]).sum())
  if var >= numpy.finfo(float).eps:
    print('Heat is not conserved!')
    sys.exit()

  if plot:
    plt.figure(figsize=(16,12))
    newSP(2,2)
    field     = 86400.0*salt_flux
    make_plot(lon,lat,field, 'Surface salt flux from ice-ocean exchange [kg m$^{-2}$ day$^{-1}$]',cmin=0.0,cmax=0.0,xlabel=True)
    nextSP()
    field     = 86400.0*salt_restore
    make_plot(lon,lat,field, 'Surface salt flux from restoring [kg m$^{-2}$ day$^{-1}$]',cmin=0.0,cmax=0.0,xlabel=True)
    plt.show()
  print('Passed!')
  return

# for easy setup of subplots
def newSP(y,x):
    global __spv, __spi ; __spv = (y,x) ; __spi = 1 ; plt.subplot(__spv[0], __spv[1], __spi)
def nextSP():
    global __spv, __spi ; __spi = __spi + 1 ; plt.subplot(__spv[0], __spv[1], __spi)


def make_plot(lon, lat, field, title, xmin=-280, xmax=80, ymin=-80, ymax=90, cmin=-200, cmax=200, xlabel=False):
   ''' Lat/Lon plot with anotations
   '''
   global area
   field_min = numpy.amin(field)
   field_max = numpy.amax(field)
   field_ave = (field*area).sum() / area.sum()
   ch = plt.pcolormesh(lon,lat,field)
   cbax=plt.colorbar(ch, extend='both')
   plt.title(r''+title)
   if (cmin != 0.0 or cmax != 0.0):
     plt.clim(cmin,cmax)

   plt.xlim(xmin,xmax)
   plt.ylim(ymin,ymax)
   plt.ylabel(r'Latitude [$\degree$N]')
   if xlabel: plt.xlabel(r'Longitude')
   axis = plt.gca()
   axis.annotate('max=%5.2f\nmin=%5.2f\nave=%5.2f'%(field_max,field_min,field_ave),xy=(0.01,0.73),
              xycoords='axes fraction', verticalalignment='bottom', fontsize=8, color='black')

   return

if __name__ == '__main__':
  run()
