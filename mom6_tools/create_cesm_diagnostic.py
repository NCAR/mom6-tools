#!/usr/bin/env python

'''
Create directory and template yaml file for a new case to be processed using mom6_tools.
'''
import os, yaml
from mom6_tools.DiagsCase import DiagsCase
import socket

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Create a new case to be processed using mom6_tools.''')
  parser.add_argument('caseroot',  type=str, help='''Path to the CASEROOT''')
  parser.add_argument('shortname', type=str, help='''A short name descring the experiment''')
  parser.add_argument('--cimeroot', type=str, default='/glade/work/gmarques/cesm.sandboxes/cesm2_3_beta08/cime/',
                     help='''Path to the CIME root used in this experiment. Default is
                     /glade/work/gmarques/cesm.sandboxes/cesm2_3_beta08/cime/''')
  parser.add_argument('-sd','--start_date', type=str, default='0038-01-01',
                      help='''Start year to compute averages. Default 0038-01-01''')
  parser.add_argument('-ed','--end_date', type=str, default='0059-01-01',
                     help='''End year to compute averages. Default 0059-01-01''')
  parser.add_argument('-p','--proj_code', type=str, default='NCGD0011',
                     help='''Project code. Default NCGD0011''')
  parser.add_argument('-ce','--conda_env', type=str, default='dev2',
                     help='''Conda env to be used for running the scripts. Default dev2''')
  parser.add_argument('-cl','--cluster', type=str, default='casper',
                     help='''Name of the cluster to run the scripts via batch jobs. Default casper, but cheyenne is also supported.''')

  parser.add_argument('-debug',   help='''Add priting statements for debugging purposes''',
                      action="store_true")
  cmdLineArgs = parser.parse_args()
  return cmdLineArgs
  main(cmdLineArgs)

def main():
  # Get options
  args = options()
  # construct a dict with essential info
  case_config = {'Avg' : {'start_date' : args.start_date,
                          'end_date' : args.end_date}}
  case_config['Case'] = {'SNAME'   : args.shortname,
                         'CASEROOT': args.caseroot,
                         'CIMEROOT': args.cimeroot}

  # extract info from args
  proj_code = args.proj_code
  cluster = args.cluster
  conda_env = args.conda_env

  # Create the case instance
  dcase = DiagsCase(case_config['Case'])
  RUNDIR = dcase.get_value('RUNDIR')
  DOUT_S_ROOT = dcase.get_value('DOUT_S_ROOT')
  if args.debug:
    print('RUNDIR:', RUNDIR)
    print('DOUT_S_ROOT:', DOUT_S_ROOT)
    print('Casename is:', dcase.casename)
  # Update dict
  OCN_DIAG_ROOT = os.getcwd() +'/'+ str(dcase.casename) + '/ncfiles/'
  #case_config['Case'].update({'RUNDIR' : RUNDIR})
  #case_config['Case'].update({'DOUT_S_ROOT' : DOUT_S_ROOT})
  case_config['Case'].update({'OCN_DIAG_ROOT' : OCN_DIAG_ROOT})
  if not os.path.isdir(dcase.casename):
    print('Creating {}... \n'.format(dcase.casename))
    os.system('mkdir -p {}/scripts'.format(dcase.casename))
    make_yaml(dcase.casename, case_config)
    # MOC
    cmd = 'moc.py diag_config.yml -nw 6 -fname .mom6.h_*.nc'
    make_PBS_batch(dcase.casename, 'moc', cmd, proj_code, cluster, conda_env)
    # PHT
    cmd = 'poleward_heat_transport.py diag_config.yml -nw 6'
    make_PBS_batch(dcase.casename, 'pht', cmd, proj_code, cluster, conda_env)
    # section transports
    cmd = 'section_transports.py diag_config.yml -save_ncfile'
    make_PBS_batch(dcase.casename, 'transports', cmd, proj_code, cluster, conda_env)
    # surface
    cmd = 'surface.py diag_config.yml -nw 6'
    make_PBS_batch(dcase.casename, 'surface', cmd, proj_code, cluster, conda_env)
    # equatorial
    cmd = 'equatorial_comparison.py diag_config.yml -nw 6'
    make_PBS_batch(dcase.casename, 'equatorial', cmd, proj_code, cluster, conda_env)
    # T&S @ vertical levels
    cmd = 'TS_levels.py diag_config.yml -nw 6'
    make_PBS_batch(dcase.casename, 'ts_levels', cmd, proj_code, cluster, conda_env)
    # stats
    cmd = 'stats.py diag_config.yml -ocean_stats -time_series'
    make_PBS_batch(dcase.casename, 'stats', cmd, proj_code, cluster, conda_env)
    # drift thetao
    cmd = 'drift.py diag_config.yml thetao --drift -nw 6'
    make_PBS_batch(dcase.casename, 'drift_thetao', cmd, proj_code, cluster, conda_env)
    # drift so
    cmd = 'drift.py diag_config.yml so --drift -nw 6'
    make_PBS_batch(dcase.casename, 'drift_so', cmd, proj_code, cluster, conda_env)
    # rms thetao
    cmd = 'drift.py diag_config.yml thetao --rms -nw 6'
    make_PBS_batch(dcase.casename, 'rms_thetao', cmd, proj_code, cluster, conda_env)
    # rms so
    cmd = 'drift.py diag_config.yml so --rms -nw 6'
    make_PBS_batch(dcase.casename, 'rms_so', cmd, proj_code, cluster, conda_env)
    # Create an ascii file to run a set of batch jobs
    make_run_script(dcase.casename)
  else:
    print('Directory {} already exists. \n'.format(dcase.casename))

  return

def make_run_script(casename):
  """
  Create an ascii file to run a set of batch jobs.
  """
  print('Writing run_scripts.sh...')
  # select job submition command based on machine
  sname = socket.gethostname()
  if 'cheyenne' in sname:
    cmd = 'qsubcasper'
  elif 'casper' in sname:
    cmd = 'qsub'
  else:
    raise ValueError("This machine not supported")

  f = open(casename+'/run_scripts.sh', 'w')
  f.write(cmd + " scripts/moc.sh \n")
  f.write(cmd + " scripts/pht.sh \n")
  f.write(cmd + " scripts/transports.sh \n")
  f.write(cmd + " scripts/surface.sh \n")
  f.write(cmd + " scripts/equatorial.sh \n")
  f.write(cmd + " scripts/ts_levels.sh \n")
  f.write(cmd + " scripts/stats.sh \n")
  f.write(cmd + " scripts/drift_thetao.sh \n")
  f.write(cmd + " scripts/drift_so.sh \n")
  f.write(cmd + " scripts/rms_thetao.sh \n")
  f.write(cmd + " scripts/rms_so.sh \n")
  f.close()
  os.system('chmod +x '+casename+'/run_scripts.sh')
  return

def make_PBS_batch(casename, diag,run_prog, proj_code, cluster, conda_env):
  """
  Create a PBS batch script with name given by diag and executes run_prog
  """
  print('Writing scripts/{}.sh...'.format(diag))
  fname = '{}/scripts/{}.sh'.format(casename, diag)
  f = open(fname, 'w')
  f.write("#!/bin/bash \n")
  f.write("#PBS -N {} \n".format(diag))
  f.write("#PBS -A {} \n".format(proj_code))
  f.write("#PBS -l select=1:ncpus=1:mem=4GB \n")
  f.write("#PBS -l walltime=02:00:00 \n")
  f.write("#PBS -q {} \n".format(cluster))
  f.write("#PBS -j oe \n")
  f.write("    \n")
  f.write("source ~/.bashrc      \n")
  f.write("conda activate {}  \n".format(conda_env))
  f.write("    \n")
  f.write("{} \n".format(run_prog))
  f.close()
  return

def make_yaml(casename, case_config):
  """
  Create a yaml file with key info.
  """
  print('Writing diag_config.yml...')
  with open(casename+'/diag_config.yml', 'w') as file:
    documents = yaml.dump(case_config, file)

  return

if __name__ == '__main__':
  main()
