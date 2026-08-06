#!/usr/bin/env python
'''
mom6-tools is a collection of scripts for working with CESM/MOM6 output.
It relies on the following python packages:
 - matplotlib
 - xarray
 - etc
'''

from importlib.metadata import version, PackageNotFoundError

#from MOM6grid import *
#from section_transports import *
#from latlon_analysis import *
#from poleward_heat_transport import *

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = None
    pass
