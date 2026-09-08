import argparse

import pytest

from mom6_tools.jobqueue import (
    add_jobqueue_args,
    cluster_class_from_name,
    get_cluster,
    release_workers,
)


# ---------------------------------------------------------------------------
# Serial (nw <= 1): no cluster of any kind should be created.
# ---------------------------------------------------------------------------

def test_serial_zero_workers():
  parallel, cluster, client = get_cluster(0)
  assert parallel is False
  assert cluster is None
  assert client is None

def test_serial_one_worker():
  # nw=1 is still "serial" -- only nw > 1 launches a cluster.
  parallel, cluster, client = get_cluster(1)
  assert parallel is False
  assert cluster is None
  assert client is None


# ---------------------------------------------------------------------------
# LocalCluster: the default when no cluster is requested, and real workers
# actually come up (this spins up an in-process dask cluster, no batch
# scheduler involved).
# ---------------------------------------------------------------------------

def test_default_cluster_is_local():
  # No cluster_class, no args: must default to LocalCluster, not PBSCluster.
  parallel, cluster, client = get_cluster(2, threads_per_worker=1)
  try:
    assert parallel is True
    assert type(cluster).__name__ == 'LocalCluster'
    assert client is not None
  finally:
    release_workers(parallel, cluster, client)

def test_explicit_local_cluster_by_string():
  parallel, cluster, client = get_cluster(2, cluster_class='LocalCluster', threads_per_worker=1)
  try:
    assert parallel is True
    assert type(cluster).__name__ == 'LocalCluster'
  finally:
    release_workers(parallel, cluster, client)

def test_explicit_local_cluster_by_class():
  from dask.distributed import LocalCluster
  parallel, cluster, client = get_cluster(2, cluster_class=LocalCluster, threads_per_worker=1)
  try:
    assert parallel is True
    assert type(cluster).__name__ == 'LocalCluster'
  finally:
    release_workers(parallel, cluster, client)

def test_local_cluster_via_cli_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-nw', '--number_of_workers', type=int, default=1)
  add_jobqueue_args(parser)
  args = parser.parse_args(['-nw', '2', '--cluster-class', 'LocalCluster'])

  parallel, cluster, client = get_cluster(args.number_of_workers, args=args, threads_per_worker=1)
  try:
    assert parallel is True
    assert type(cluster).__name__ == 'LocalCluster'
  finally:
    release_workers(parallel, cluster, client)


# ---------------------------------------------------------------------------
# cluster_class_from_name(): resolution rules on their own.
# ---------------------------------------------------------------------------

def test_cluster_class_from_name_none_is_local():
  from dask.distributed import LocalCluster
  assert cluster_class_from_name(None) is LocalCluster
  assert cluster_class_from_name('') is LocalCluster

def test_cluster_class_from_name_unknown_raises():
  with pytest.raises(ValueError):
    cluster_class_from_name('NotARealCluster')


# ---------------------------------------------------------------------------
# Batch clusters are opt-in: resource settings alone never select one, and
# never get silently dropped.
# ---------------------------------------------------------------------------

JOBQUEUE_YAML = {
    'cluster_class': 'PBSCluster',
    'cores': 1,
    'memory': '4GB',
    'queue': 'casper',
    'resource_spec': 'select=1:ncpus=1:mem=4GB',
}


def test_cluster_class_from_yaml_block():
  # cluster_class: PBSCluster in the Jobqueue: block is what opts in.
  dask_jobqueue = pytest.importorskip('dask_jobqueue')
  captured = {}

  class FakePBSCluster:
    job_cls = object()
    dashboard_link = 'fake'

    def __init__(self, **kw):
      captured.update(kw)

    def scale(self, n):
      captured['scaled'] = n

  import dask.distributed
  monkey = dask.distributed.Client
  dask.distributed.Client = lambda cluster: 'fake-client'
  try:
    parallel, cluster, client = get_cluster(
        4, cluster_class=FakePBSCluster, config=JOBQUEUE_YAML)
  finally:
    dask.distributed.Client = monkey

  assert parallel is True
  assert captured['scaled'] == 4
  assert captured['queue'] == 'casper'
  assert captured['resource_spec'] == 'select=1:ncpus=1:mem=4GB'
  # cluster_class is a mom6_tools setting, not a cluster kwarg.
  assert 'cluster_class' not in captured


def test_yaml_cluster_class_is_resolved_by_name():
  dask_jobqueue = pytest.importorskip('dask_jobqueue')
  from mom6_tools.jobqueue import cluster_class_from_name
  assert cluster_class_from_name(
      JOBQUEUE_YAML['cluster_class']) is dask_jobqueue.PBSCluster


def test_config_is_never_read_off_args():
  # Scripts must pass the Jobqueue: block as config=; smuggling it through
  # the argparse namespace is not a supported path and must not work.
  args = argparse.Namespace(jobqueue_config=JOBQUEUE_YAML)
  parallel, cluster, client = get_cluster(2, args=args, threads_per_worker=1)
  try:
    assert type(cluster).__name__ == 'LocalCluster'
  finally:
    release_workers(parallel, cluster, client)


def test_jobqueue_settings_without_cluster_class_warn():
  # A Jobqueue: block missing cluster_class must not silently vanish.
  config = {k: v for k, v in JOBQUEUE_YAML.items() if k != 'cluster_class'}
  with pytest.warns(UserWarning, match='Ignoring Jobqueue settings'):
    parallel, cluster, client = get_cluster(2, config=config, threads_per_worker=1)
  try:
    assert type(cluster).__name__ == 'LocalCluster'
  finally:
    release_workers(parallel, cluster, client)


def test_cli_jobqueue_flags_without_cluster_class_warn():
  parser = argparse.ArgumentParser()
  parser.add_argument('-nw', '--number_of_workers', type=int, default=1)
  add_jobqueue_args(parser)
  args = parser.parse_args(['-nw', '2', '--queue', 'casper', '--cores', '4'])

  with pytest.warns(UserWarning, match='cores, queue'):
    parallel, cluster, client = get_cluster(
        args.number_of_workers, args=args, threads_per_worker=1)
  try:
    assert type(cluster).__name__ == 'LocalCluster'
  finally:
    release_workers(parallel, cluster, client)


def test_unrecognized_jobqueue_key_warns():
  # Keys the whitelist drops (e.g. job_extra_directives) must be reported.
  config = dict(JOBQUEUE_YAML, job_extra_directives=['-l gpu_type=v100'])
  del config['cluster_class']
  with pytest.warns(UserWarning, match='job_extra_directives'):
    parallel, cluster, client = get_cluster(2, config=config, threads_per_worker=1)
  release_workers(parallel, cluster, client)


def test_no_warning_when_no_jobqueue_settings():
  # Only our own warnings are checked here; dask emits unrelated ones (e.g.
  # about the dashboard port already being in use).
  import warnings as _warnings
  with _warnings.catch_warnings(record=True) as caught:
    _warnings.simplefilter('always')
    parallel, cluster, client = get_cluster(2, threads_per_worker=1)
    release_workers(parallel, cluster, client)
  assert not [w for w in caught if 'Jobqueue' in str(w.message)]
