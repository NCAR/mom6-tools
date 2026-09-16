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

3. Register ``mom6-tools`` in ``ipykernel``::

    conda run -n mom6-tools python -m ipykernel install --user --name mom6-tools


Notes 
----------------------------

1. ``get_cluster(n_workers)`` runs a ``dask.distributed.LocalCluster`` on the node it is called from. Batch workers are opt-in: to submit them you must name the class, either with ``parallel, cluster, client = get_cluster(n_workers, cluster_class='PBSCluster')``, with ``--cluster-class PBSCluster`` on the command line, or with ``cluster_class: PBSCluster`` in the ``Jobqueue:`` block of ``diag_config.yml``. Resource settings on their own (``--queue``, a ``Jobqueue:`` block without ``cluster_class``, ...) will not do it -- they are ignored, with a warning naming them.
2. If running on HPC, make sure the project account is added to ``~/.config/dask/jobqueue.yaml``. Or set a default account as an environment variable that your scheduler can access. Anything not set in the ``Jobqueue:`` block or on the command line falls back to that dask configuration.

