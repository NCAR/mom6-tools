Installing
==========

You can install mom6-tools with ``pip``, ``conda``, or by installing from source.

Pip
---

Pip can be used to install mom6-tools::

   pip install mom6-tools

Conda
-----

To install the latest version of mom6-tools from the
`conda-forge <https://conda-forge.github.io/>`_ repository using
`conda <https://www.anaconda.com/downloads>`_::

    conda install -c conda-forge mom6-tools

Install from Source
-------------------

To install mom6-tools from source, clone the repository from `github
<https://github.com/NCAR/mom6-tools>`_::

    git clone https://github.com/NCAR/mom6-tools.git
    cd mom6-tools
    pip install -e .

You can also install directly from git master branch::

    pip install git+https://github.com/NCAR/mom6-tools


Test
----

To run mom6-tools's tests with ``pytest``::

    git clone https://github.com/NCAR/mom6-tools.git
    cd mom6-tools
    pytest - v
