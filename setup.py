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
)
