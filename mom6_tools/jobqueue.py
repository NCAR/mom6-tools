"""Helpers for launching dask-jobqueue clusters.

This module replaces the previous dependency on NCAR's `ncar_jobqueue`
package. It provides default PBS resource configurations for NCAR's HPC
systems (Derecho, Casper) while remaining usable on any other HPC system:
outside of a recognized NCAR host, cluster
construction falls back to plain `dask_jobqueue` behavior, which reads
its configuration from the user's own dask config files (e.g.
`~/.config/dask/jobqueue.yaml`). See
https://jobqueue.dask-jobqueue.org/en/latest/configuration.html
"""

import getpass
import re
import socket

import dask_jobqueue
from dask.distributed import Client

# hostname patterns used to identify NCAR's HPC systems
_NCAR_HOSTS = [
    ('derecho', re.compile(r'^(derecho|dec)\d+')),
    ('casper', re.compile(r'^(casper|crhtc|crlogin)\d+')),
]

# Default dask_jobqueue.PBSCluster keyword arguments for each NCAR system.
# {user} is filled in with the current username. Any of these can be
# overridden by passing keyword arguments to get_cluster().
NCAR_PBS_DEFAULTS = {
    'derecho': dict(
        cores=1, memory='4GiB', processes=1, interface='hsn0', queue='develop',
        walltime='01:00:00', resource_spec='select=1:ncpus=1:mem=4GB',
        job_extra_directives=['-l job_priority=economy', '-r n'],
        log_directory='/glade/derecho/scratch/{user}/dask/derecho/logs',
        local_directory='/glade/derecho/scratch/{user}/dask/derecho/local-dir',
        death_timeout=60,
    ),
    'casper': dict(
        cores=1, memory='4GiB', processes=1, interface='ext', queue='casper',
        walltime='01:00:00', resource_spec='select=1:ncpus=1:mem=4GB',
        job_extra_directives=['-r n'],
        log_directory='/glade/derecho/scratch/{user}/dask/casper/logs',
        local_directory='/glade/derecho/scratch/{user}/dask/casper/local-dir',
        death_timeout=60,
    ),
}


def identify_host():
  '''Return the name of the NCAR HPC system this code is running on
  (of 'derecho', 'casper'), or None if the host is not a recognized
  NCAR system.
  '''
  hostname = socket.getfqdn()
  for name, regex in _NCAR_HOSTS:
    if regex.search(hostname):
      return name
  return None


def get_cluster(n_workers, cluster_class=dask_jobqueue.PBSCluster, **kwargs):
  '''Launch a dask_jobqueue cluster and client with n_workers workers.

  On a recognized NCAR HPC system (i.e. Casper or Derecho), default PBS resource settings
  are applied automatically. On any other system, dask_jobqueue's normal
  configuration discovery is used (e.g. ~/.config/dask/jobqueue.yaml), and
  a different scheduler (e.g. dask_jobqueue.SLURMCluster) can be selected
  via cluster_class. Any keyword arguments passed here override the
  defaults for the current host.

  Parameters
  ----------
  n_workers : int
    Number of workers to request.
  cluster_class : dask_jobqueue cluster class, optional
    Defaults to dask_jobqueue.PBSCluster (used by all of NCAR's systems).

  Returns
  -------
  cluster, client
  '''
  config = {}
  if cluster_class is dask_jobqueue.PBSCluster: # identify and config NCAR system defaults
    host = identify_host()
    if host is not None:
      user = getpass.getuser()
      config = {
        key: (value.format(user=user) if key in ('log_directory', 'local_directory') else value)
        for key, value in NCAR_PBS_DEFAULTS[host].items()
      }
  
  # Set kwargs and launch cluster and client
  config.update(kwargs)
  cluster = cluster_class(**config)
  cluster.scale(n_workers)
  client = Client(cluster)
  return cluster, client
