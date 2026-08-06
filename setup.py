#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys
from os.path import exists

from setuptools import setup

if exists('README.md'):
    with open('README.md') as f:
        long_description = f.read()
else:
    long_description = ''

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

test_requirements = ['pytest']


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
]

setup(
    maintainer='Gustavo Marques',
    maintainer_email='gmarques@ucar.edu',
    description='MOM6-CESM tools',
    install_requires=install_requires,
    license='Apache License 2.0',
    long_description=long_description,
    classifiers=CLASSIFIERS,
    keywords='ocean modeling',
    name='mom6-tools',
    packages=['mom6_tools'],
    test_suite='tests',
    tests_require=test_requirements,
    include_package_data=True,
    url='https://github.com/NCAR/mom6-tools',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0'],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'mom6-tools_aaiw_pv=mom6_tools.aaiw_pv:main',
            'mom6-tools_buoyancy_flux=mom6_tools.buoyancy_flux:main',
            'mom6-tools_compute_basin_reductions=mom6_tools.compute_basin_reductions:main',
            'mom6-tools_create_cesm_diagnostics=mom6_tools.create_cesm_diagnostics:main',
            'mom6-tools_create_climatology=mom6_tools.create_climatology:main',
            'mom6-tools_create_mom6_tools=mom6_tools.create_mom6_tools:main',
            'mom6-tools_diff_rms=mom6_tools.diff_rms:main',
            'mom6-tools_drift=mom6_tools.drift:main',
            'mom6-tools_enso=mom6_tools.enso:main',
            'mom6-tools_equatorial_comparison=mom6_tools.equatorial_comparison:main',
            'mom6-tools_forcing=mom6_tools.forcing:main',
            'mom6-tools_latlon_analysis=mom6_tools.latlon_analysis:main',
            'mom6-tools_moc=mom6_tools.moc:main',
            'mom6-tools_moc_sigma2=mom6_tools.moc_sigma2:main',
            'mom6-tools_mom6_xyplot=mom6_tools.mom6_xyplot:main',
            'mom6-tools_poleward_heat_transport=mom6_tools.poleward_heat_transport:main',
            'mom6-tools_section_transports=mom6_tools.section_transports:main',
            'mom6-tools_stats=mom6_tools.stats:main',
            'mom6-tools_surface=mom6_tools.surface:main',
            'mom6-tools_tao_mooring_comparison=mom6_tools.tao_mooring_comparison:main',
            'mom6-tools_TS_levels=mom6_tools.TS_levels:main',
            'mom6-tools_verticalvelocity=mom6_tools.verticalvelocity:main',
            'mom6-tools_wind_stress=mom6_tools.wind_stress:main',
        ]
    }
)
