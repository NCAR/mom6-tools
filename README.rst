.. image:: https://github.com/NCAR/mom6-tools/actions/workflows/python-tests.yml/badge.svg
    :target: https://github.com/NCAR/mom6-tools/actions/workflows/python-tests.yml
    :alt: Tests Status

.. image:: https://img.shields.io/readthedocs/mom6-tools/latest.svg?style=flat - default
    :target: https://mom6-tools.readthedocs.io/?badge=latest
    :alt: Documentation Status

Tools to support analysis of CESM/MOM6 model solutions. See
documentation_ for more information.

.. _documentation: https://mom6-tools.readthedocs.io/


Installation 
----------------------------

1. Clone the repository from `github <https://github.com/NCAR/mom6-tools>`_::

    git clone https://github.com/NCAR/mom6-tools.git

2. Create a new conda environment and install required packages::

    cd mom6-tools
    conda env create --file environment.yml

3. Install MOM6-tools::

    conda activate mom6-tools
    python setup.py install

4. Add ``mom6-tools/mom6_tools`` to path, e.g., add this line to ``~/.bashrc``::

    export PATH=$PATH:/glade/work/${USER}/mom6-tools/mom6_tools/

5. Make sure the project account is added to ``~/.config/dask/ncar-jobqueue.yaml``

6. Optional: run ``pytest``

