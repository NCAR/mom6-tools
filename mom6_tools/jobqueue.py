"""Helper for launching a dask cluster: a plain dask.distributed.LocalCluster
running on the current node (the default -- no batch job submission), or a
dask_jobqueue batch cluster (e.g. PBSCluster, SLURMCluster) when explicitly
requested.

get_cluster() does this in three steps:

1. Collect cluster_class and config from every source: cluster_class as
   given, else args.cluster_class from a parser that called
   add_jobqueue_args(), else a cluster_class: key inside config -- the
   YAML Jobqueue: block, which callers pass in explicitly as config=. A
   batch cluster is always opt-in and is never inferred from the presence
   of resource settings: submitting batch jobs requires naming the class
   (e.g. PBSCluster) in one of those three places.
2. Process: resolve cluster_class to an actual class via
   cluster_class_from_name(). For a dask_jobqueue class, also resolve its
   resource kwargs (cores, memory, queue, walltime, ...) via
   resolve_jobqueue_kwargs() -- CLI flags on args, then the YAML Jobqueue:
   block, then dask's own configuration discovery (e.g.
   ~/.config/dask/jobqueue.yaml, see
   https://jobqueue.dask.org/en/latest/configuration.html). Explicit
   **kwargs given to get_cluster() always win over anything resolved this
   way. There are no mom6_tools-specific defaults beyond that -- if a
   required setting (e.g. cores/memory) is still unset, cluster_class
   raises its own clear error; site-specific defaults belong in a
   diag_config.yml Jobqueue: block.
3. Launch the cluster and wrap it in a dask.distributed.Client.

resolve_jobqueue_kwargs() does not apply to LocalCluster (or anything else
without a dask_jobqueue job_cls); pass its kwargs (e.g. threads_per_worker,
memory_limit) directly to get_cluster() instead. Resource settings supplied
while LocalCluster is in effect are ignored with a warning naming them, as
are keys in the Jobqueue: block that are not in JOBQUEUE_CONFIG_KEYS.
"""

import warnings
import argparse

# Settings shared by the diag_config.yml `Jobqueue:` block, the CLI flags
# from add_jobqueue_args(), and the kwargs accepted by dask_jobqueue cluster
# classes (e.g. PBSCluster, SLURMCluster).
JOBQUEUE_KEYS = [
    'cores', 'memory', 'processes', 'interface', 'queue', 'walltime',
    'resource_spec', 'account', 'log_directory', 'local_directory',
]

# Everything the `Jobqueue:` block may contain. It also accepts
# cluster_class, so a diag_config.yml can opt into batch workers (e.g.
# cluster_class: PBSCluster) without every caller passing --cluster-class.
JOBQUEUE_CONFIG_KEYS = ['cluster_class'] + JOBQUEUE_KEYS


def add_jobqueue_args(parser):
  '''Add the standard dask-jobqueue CLI flags to an argparse parser.

  Each flag defaults to None so resolve_jobqueue_kwargs() can distinguish a
  value the user explicitly passed from one picked up elsewhere.
  '''
  group = parser.add_argument_group('dask-jobqueue')
  group.add_argument('--cores', type=int, default=None, help='Cores per worker job.')
  group.add_argument('--memory', type=str, default=None, help="Memory per worker job, e.g. '4GB'.")
  group.add_argument('--processes', type=int, default=None, help='Python processes per worker job.')
  group.add_argument('--interface', type=str, default=None, help="Network interface, e.g. 'ib0'.")
  group.add_argument('--queue', type=str, default=None, help='Batch queue to submit worker jobs to.')
  group.add_argument('--walltime', type=str, default=None, help="Walltime per worker job, e.g. '02:00:00'.")
  group.add_argument('--resource-spec', type=str, default=None, help='PBS resource_spec string.')
  group.add_argument('--account', type=str, default=None, help='Account/project to charge worker jobs to.')
  group.add_argument('--log-directory', type=str, default=None, help='Directory for worker job logs.')
  group.add_argument('--local-directory', type=str, default=None, help='Directory for worker scratch space.')
  group.add_argument('--cluster-class', type=str, default=None,
                      help="Exact dask cluster class name to use, e.g. 'LocalCluster' "
                           "(default, runs workers on this node without submitting batch "
                           "jobs), 'PBSCluster', or 'SLURMCluster'. dask_jobqueue "
                           "classes: https://jobqueue.dask.org/en/latest/api.html "
                           "-- LocalCluster: "
                           "https://docs.dask.org/en/stable/deploying-python.html")
  return parser


def cluster_class_from_name(cluster_class):
  '''Resolve cluster_class to an actual class, for get_cluster().

  cluster_class may already be a class (returned unchanged), a class name
  string such as 'PBSCluster' or 'SLURMCluster' (looked up in
  dask_jobqueue; see https://jobqueue.dask.org/en/latest/api.html) or
  'LocalCluster' explicitly, or None/'' (resolves to
  dask.distributed.LocalCluster, the default when no specific batch
  cluster is requested; see
  https://docs.dask.org/en/stable/deploying-python.html).
  '''
  # Already resolved: a dask_jobqueue cluster class, or LocalCluster pass
  if isinstance(cluster_class, type):
    if hasattr(cluster_class, 'job_cls'):
      return cluster_class
    try:
      from dask.distributed import LocalCluster
      if issubclass(cluster_class, LocalCluster):
        return cluster_class
    except ImportError:
      pass

  # None or '': no cluster requested - default to LocalCluster.
  if not cluster_class or cluster_class == 'LocalCluster':
    from dask.distributed import LocalCluster
    return LocalCluster

  # Otherwise, cluster_class must be a class name string to look up.
  if not isinstance(cluster_class, str):
    raise ValueError(
        "cluster_class must be a dask_jobqueue cluster class, "
        "dask.distributed.LocalCluster, a class name string, or None; "
        "got {!r}.".format(cluster_class))

  try:  
    import dask_jobqueue  
  except ImportError:  
    raise ValueError(  
        "Cannot resolve cluster_class {!r}: dask_jobqueue is not installed. "  
        "Install it (e.g. `conda install dask-jobqueue`), or use "  
        "'LocalCluster' to run workers on this node without submitting batch "  
        "jobs.".format(cluster_class))

  resolved_class = getattr(dask_jobqueue, cluster_class, None)
  if resolved_class is not None:
    return resolved_class

  raise ValueError(
      "Unknown --cluster-class {!r}. Pass the exact class name as it "
      "appears in dask_jobqueue (e.g. 'PBSCluster', 'SLURMCluster' -- see "
      "https://jobqueue.dask.org/en/latest/api.html), or "
      "'LocalCluster' to run workers on this node without submitting batch "
      "jobs (see https://docs.dask.org/en/stable/deploying-python.html)."
      .format(cluster_class))


def resolve_jobqueue_kwargs(cluster_class, args={}, config={}):
  '''Merge CLI and YAML resource kwargs for an already-resolved
  dask_jobqueue cluster_class.

  Only the CLI (args) and the YAML Jobqueue: block (config) are merged
  here, CLI taking priority.

  Parameters
  ----------
  cluster_class : dask_jobqueue cluster class
    An already-resolved dask_jobqueue class (e.g. dask_jobqueue.PBSCluster
    or dask_jobqueue.SLURMCluster) -- see cluster_class_from_name(). Not
    for dask.distributed.LocalCluster.
  args : argparse.Namespace, optional
    Parsed CLI args, e.g. from a parser that called add_jobqueue_args().
  config : dict, optional
    The `Jobqueue:` block from diag_config.yml (or any dict of the same
    keys as JOBQUEUE_KEYS).

  Returns
  -------
  dict of kwargs, ready to pass to cluster_class(**kwargs)
  '''
  if not hasattr(cluster_class, 'job_cls'):
    raise ValueError(
        "resolve_jobqueue_kwargs() only supports dask_jobqueue cluster "
        "classes (e.g. PBSCluster, SLURMCluster), got {!r}. For a plain "
        "dask.distributed.LocalCluster, pass kwargs directly to "
        "get_cluster() instead.".format(cluster_class))
    
  if isinstance(args, argparse.Namespace):
      args = vars(args)
  elif not isinstance(args, dict):
    raise ValueError(f"args must be an argparse.Namespace or dict, got {type(args).__name__}.")

  kwargs = {}
  for key in JOBQUEUE_KEYS:
    cli_value = args.get(key)
    if cli_value is not None:  
      kwargs[key] = cli_value  
    else:  
      kwargs[key] = config.get(key) 
  return kwargs


def _jobqueue_settings_given(args={}, config={}):
  '''Names of the JOBQUEUE_KEYS settings actually supplied via CLI or YAML.

  Used to warn when resource settings were given but the resolved cluster
  class is not a dask_jobqueue one and so will ignore them.
  '''
  if isinstance(args, argparse.Namespace):
      args = vars(args)
  elif not isinstance(args, dict):
    raise ValueError(f"args must be an argparse.Namespace or dict, got {type(args).__name__}.")
  
  given = []
  for key in JOBQUEUE_KEYS:
    if args.get(key) is not None or config.get(key) is not None:  
      given.append(key) 
  return given


def get_cluster(nw, cluster_class=None, args={}, config={}, **kwargs):
  '''Request nw dask workers, or run in serial if nw <= 1.

  This is the single entry point for launching a dask cluster, for both
  interactive/notebook use and for scripts driven by argparse + a YAML
  diag_config.yml. See the module docstring for more info. 

  - Interactively, on this node: get_cluster(6), or
    get_cluster(6, threads_per_worker=1).
  - Interactively, submitting batch jobs -- cluster_class is required, the
    resource kwargs alone will not do it: get_cluster(40,
    cluster_class='PBSCluster', cores=4, processes=1,
    resource_spec='select=1:ncpus=1:mem=10GB'), or
    get_cluster(nw, cluster_class='SLURMCluster', queue='regular').
  - From a script: get_cluster(nw, args=args,
    config=diag_config_yml.get('Jobqueue')), where args came from a parser
    that called add_jobqueue_args(). The YAML block is only ever read from
    config -- it is never picked up off args -- and a cluster_class: key
    in it counts as naming the class.

  If nw <= 1, or if the required modules can't be imported, no cluster is
  created and this returns parallel=False.

  Parameters
  ----------
  nw : int
    Number of workers to request.
  cluster_class : class, str, or None
    See cluster_class_from_name(). None (the default) falls back to
    args.cluster_class if set, then to config['cluster_class'] if set,
    else dask.distributed.LocalCluster.
  args : argparse.Namespace, optional
    Parsed CLI args, e.g. from a parser that called add_jobqueue_args().
  config : dict, optional
    The `Jobqueue:` block from diag_config.yml, if any.
  **kwargs
    Passed directly to cluster_class, taking priority over anything
    resolved from args/config.

  Returns
  -------
  parallel, cluster, client
  '''
  if nw <= 1:
    print('No workers requested \n')
    return False, None, None
  
  if isinstance(args, argparse.Namespace):
    args = vars(args)
  elif not isinstance(args, dict):
    raise ValueError(f"args must be an argparse.Namespace or dict, got {type(args).__name__}.")

  # 1. Collect cluster_class and config from every source.
  unknown = [key for key in config if key not in JOBQUEUE_CONFIG_KEYS]
  if unknown:
    warnings.warn(
        "Ignoring unrecognized Jobqueue settings: {}. Recognized settings "
        "are: {}.".format(', '.join(sorted(unknown)),
                          ', '.join(JOBQUEUE_CONFIG_KEYS)))
  if cluster_class is None:
    cluster_class = args.get('cluster_class')

  # 2. Process: resolve the cluster class and, for a dask_jobqueue class,
  # its resource kwargs.
  try:
    import dask
    from dask.distributed import Client
    cluster_class = cluster_class_from_name(cluster_class)
  except ImportError:
    warnings.warn("Unable to import dask or dask.distributed. The script will \
           run in serial. Please install these modules, or pass an \
           already-imported cluster_class (e.g. dask.distributed.LocalCluster), if \
           you want to run in parallel.")
    print('No workers requested. \n')
    return False, None, None

  is_jobqueue_cluster = hasattr(cluster_class, 'job_cls')
  if is_jobqueue_cluster:
    resolved_kwargs = resolve_jobqueue_kwargs(cluster_class, args=args, config=config)
    resolved_kwargs.update(kwargs)
    kwargs = resolved_kwargs
  else:
    ignored = _jobqueue_settings_given(args=args, config=config)
    if ignored:
      warnings.warn(
          "Ignoring Jobqueue settings ({}) -- the resolved cluster class is "
          "{}, which does not accept them. Set `cluster_class: PBSCluster` in "
          "the diag_config.yml Jobqueue: block, or pass "
          "--cluster-class PBSCluster, if you want batch workers."
          .format(', '.join(ignored), cluster_class.__name__))

  # 3. Spin up the cluster.
  print('Requesting {} workers... \n'.format(nw))
  if is_jobqueue_cluster:
    dask.config.set({'distributed.dashboard.link': '/proxy/{port}/status'})
    cluster = cluster_class(**kwargs)
    cluster.scale(nw)
  else:
    cluster = cluster_class(n_workers=nw, **kwargs)
  client = Client(cluster)
  print(cluster.dashboard_link)
  return True, cluster, client


def release_workers(parallel, cluster, client):
  '''Close the client/cluster returned by get_cluster(), if one was started.'''
  if parallel:
    print('Releasing workers.')
    client.close()
    cluster.close()
