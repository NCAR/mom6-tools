#!/usr/bin/env python
'''
mom6-tools is a collection of scripts for working with CESM/MOM6 output.
It relies on the following python packages:
 - matplotlib
 - xarray
 - etc
'''

from importlib.metadata import PackageNotFoundError, version

#from MOM6grid import *
#from section_transports import *
#from latlon_analysis import *
#from poleward_heat_transport import *

try:
    __version__ = version('mom6-tools')
except PackageNotFoundError:
    # package is not installed
    pass
