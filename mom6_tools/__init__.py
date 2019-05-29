#!/usr/bin/env python
'''
mom6-tools is a collection of scripts for working with CESM/MOM6 output.
It relies on the following python packages:
 - matplotlib
 - xarray
 - etc
'''

from pkg_resources import DistributionNotFound, get_distribution

from .MOM6grid import *
from .section_transports import *
from .latlon_analysis import *
from .poleward_heat_transport import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
