import pytest

from mom6_tools.cli import get_parser
from mom6_tools.jobqueue import JOBQUEUE_CONFIG_KEYS, resolve_jobqueue_kwargs


def test_shared_options_parse():
  args = get_parser('test').parse_args(
      ['diag_config.yml', '-sd', '0001-01-01', '-ed', '0010-01-01', '-nw', '4',
       '-debug', '--queue', 'casper', '--cluster-class', 'PBSCluster'])
  assert args.diag_config_yml_path == 'diag_config.yml'
  assert (args.start_date, args.end_date) == ('0001-01-01', '0010-01-01')
  assert args.number_of_workers == 4
  assert args.debug is True
  assert args.queue == 'casper'
  assert args.cluster_class == 'PBSCluster'

def test_shared_option_defaults():
  args = get_parser('test').parse_args(['diag_config.yml'])
  assert (args.start_date, args.end_date) == ('', '')
  assert args.number_of_workers == 0
  assert args.debug is False
  # jobqueue flags stay None so the YAML Jobqueue: block / dask config can fill them.
  assert args.cores is None and args.cluster_class is None

def test_optional_groups_can_be_dropped():
  args = get_parser('test', config=False, dates=False, workers=False).parse_args([])
  assert vars(args) == {'debug': False}

def test_script_positional_follows_config():
  # e.g. drift.py adds 'var' after the shared diag_config_yml_path.
  parser = get_parser('test', dates=False)
  parser.add_argument('var', type=str)
  args = parser.parse_args(['diag_config.yml', 'thetao'])
  assert (args.diag_config_yml_path, args.var) == ('diag_config.yml', 'thetao')

def test_jobqueue_flags_match_jobqueue_keys():
  # The flags are defined in cli.py but read by jobqueue.py via JOBQUEUE_CONFIG_KEYS.
  args = vars(get_parser('test', config=False, dates=False).parse_args([]))
  jobqueue_args = {k: v for k, v in args.items() if k not in ('debug', 'number_of_workers')}
  assert set(jobqueue_args) == set(JOBQUEUE_CONFIG_KEYS)
  assert all(v is None for v in jobqueue_args.values())

def test_cli_flags_override_yaml_jobqueue_block():
  dask_jobqueue = pytest.importorskip('dask_jobqueue')
  args = get_parser('test').parse_args(['diag_config.yml', '--queue', 'develop', '--cores', '8'])
  config = {'cluster_class': 'PBSCluster', 'queue': 'casper', 'cores': 1, 'memory': '4GB'}
  kwargs = resolve_jobqueue_kwargs(dask_jobqueue.PBSCluster, args=args, config=config)
  assert kwargs['queue'] == 'develop'
  assert kwargs['cores'] == 8
  assert kwargs['memory'] == '4GB'
