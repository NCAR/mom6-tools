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

2. Install required packages and ``mom6-tools`` in a brand new conda environment::

    cd mom6-tools
    conda env create --file environment.yml

3. Add ``mom6-tools/mom6_tools`` to path, e.g., add this line to ``~/.bashrc``::

    export PATH=$PATH:/glade/work/${USER}/mom6-tools/mom6_tools/

4. Register ``mom6-tools`` in ``ipykernel``::

    conda run -n mom6-tools python -m ipykernel install --user --name mom6-tools


Notes 
----------------------------

1. Make sure the project account is added to ``~/.config/dask/ncar-jobqueue.yaml``.

2. If running on Casper, change ``casper`` to ``casper-dav`` in ``~/.config/dask/ncar-jobqueue.yaml``.

