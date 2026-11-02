"""Shared argparse setup for the mom6_tools command-line scripts.

get_parser() returns a parser with the options most scripts have in common;
each script then adds its own arguments and calls parse_args().
"""

import argparse


def get_parser(description, epilog=None, config=True, dates=True, workers=True):
  '''Return an ArgumentParser with the shared mom6_tools options.

  Parameters
  ----------
  description, epilog : str
    Passed to argparse.ArgumentParser.
  config : bool
    Add the diag_config_yml_path positional argument.
  dates : bool
    Add -sd/--start_date and -ed/--end_date.
  workers : bool
    Add -nw/--number_of_workers and the dask-jobqueue flags (--cores,
    --memory, --queue, --cluster-class, ...) read by jobqueue.get_cluster().

  -debug is always added.
  '''
  parser = argparse.ArgumentParser(description=description, epilog=epilog)
  if config:
    parser.add_argument('diag_config_yml_path', type=str,
                        help='''Relative or absolute path to the yaml file describing the run and diagnostics to be performed.''')
  if dates:
    parser.add_argument('-sd', '--start_date', type=str, default='',
                        help='''Start date to compute averages. Default is to use value set in the yaml file.''')
    parser.add_argument('-ed', '--end_date', type=str, default='',
                        help='''End date to compute averages. Default is to use value set in the yaml file.''')
  parser.add_argument('-debug', action='store_true',
                      help='''Add printing statements for debugging purposes.''')
  if workers:
    parser.add_argument('-nw', '--number_of_workers', type=int, default=0,
                        help='''Number of workers to use (default=0; 0 or 1 runs in serial).''')
    # dask-jobqueue flags: dests match jobqueue.JOBQUEUE_CONFIG_KEYS, and each defaults to None so
    # jobqueue.resolve_jobqueue_kwargs() can tell a value passed here from one set in the yaml file.
    group = parser.add_argument_group('dask-jobqueue')
    group.add_argument('--cores', type=int,
                       help='''Cores per worker job.''')
    group.add_argument('--memory', type=str,
                       help='''Memory per worker job, e.g. '4GB'.''')
    group.add_argument('--processes', type=int,
                       help='''Python processes per worker job.''')
    group.add_argument('--interface', type=str,
                       help='''Network interface, e.g. 'ib0'.''')
    group.add_argument('--queue', type=str,
                       help='''Batch queue to submit worker jobs to.''')
    group.add_argument('--walltime', type=str,
                       help='''Walltime per worker job, e.g. '02:00:00'.''')
    group.add_argument('--resource-spec', type=str,
                       help='''PBS resource_spec string.''')
    group.add_argument('--account', type=str,
                       help='''Account/project to charge worker jobs to.''')
    group.add_argument('--log-directory', type=str,
                       help='''Directory for worker job logs.''')
    group.add_argument('--local-directory', type=str,
                       help='''Directory for worker scratch space.''')
    group.add_argument('--cluster-class', type=str,
                       help='''Exact dask cluster class name to use, e.g. 'LocalCluster' (default, runs workers on
                       this node without submitting batch jobs), 'PBSCluster', or 'SLURMCluster'.
                       dask_jobqueue classes: https://jobqueue.dask.org/en/latest/api.html
                       -- LocalCluster: https://docs.dask.org/en/stable/deploying-python.html''')
  return parser
