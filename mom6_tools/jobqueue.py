"""Helper for launching dask-jobqueue clusters.

Resource settings (queue, walltime, cores, memory, etc.) come from either:

- dask's own configuration discovery, e.g. ``~/.config/dask/jobqueue.yaml``
  (see https://jobqueue.dask-jobqueue.org/en/latest/configuration.html), or
- keyword arguments passed explicitly to ``get_cluster()``, e.g. sourced
  from a ``Jobqueue:`` block in ``diag_config.yml``, or set manually in a
  notebook.
"""

import warnings


def get_cluster(nw, cluster_class=None, **kwargs):
  '''Request nw dask-jobqueue workers, or run in serial if nw <= 1.

  If nw > 1, launches a dask_jobqueue cluster and client with nw workers and
  returns parallel=True. If nw <= 1, or if dask_jobqueue/dask can't be
  imported, no cluster is created and this returns parallel=False.

  Parameters
  ----------
  nw : int
    Number of workers to request.
  cluster_class : dask_jobqueue cluster class, optional
    Defaults to dask_jobqueue.PBSCluster. Pass e.g. dask_jobqueue.SLURMCluster
    to target a different scheduler.
  **kwargs
    Passed directly to cluster_class, overriding any values found via dask's
    configuration discovery.

  Returns
  -------
  parallel, cluster, client
  '''
  if nw > 1:
    try:
      import dask
      import dask_jobqueue
      from dask.distributed import Client
    except ImportError:
      nw = 0
      warnings.warn("Unable to import the following: dask_jobqueue, dask and dask.distributed. \
             The script will run in serial. Please install these modules if you want \
             to run in parallel.")

  if nw > 1:
    if cluster_class is None:
      cluster_class = dask_jobqueue.PBSCluster
    print('Requesting {} workers... \n'.format(nw))
    dask.config.set({'distributed.dashboard.link': '/proxy/{port}/status'})
    cluster = cluster_class(**kwargs)
    cluster.scale(nw)
    client = Client(cluster)
    print(cluster.dashboard_link)
    parallel = True
  else:
    print('No workers requested... \n')
    parallel = False
    cluster = None
    client = None
  return parallel, cluster, client
