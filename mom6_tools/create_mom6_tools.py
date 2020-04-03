#!/usr/bin/env python

'''
Create directory and template yaml file for a new case to be processed using mom6_tools.
'''
import os, yaml
from mom6_tools.DiagsCase import DiagsCase

def options():
  try: import argparse
  except: raise Exception('This version of python is not new enough. python 2.7 or newer is required.')
  parser = argparse.ArgumentParser(description='''Create a new case to be processed using mom6_tools.''')
  parser.add_argument('caseroot', type=str, help='''Path to the CASEROOT''')
  parser.add_argument('--cimeroot', type=str, default='/glade/work/gmarques/cesm.sandboxes/cesm2_2_alpha04b_mom6/cime',
                     help='''Path to the CIME root used in this experiment. Default is
                     /glade/work/gmarques/cesm.sandboxes/cesm2_2_alpha04b_mom6/cime''')
  parser.add_argument('-sd','--start_date', type=str, default='0038-01-01',
                      help='''Start year to compute averages. Default 0038-01-01''')
  parser.add_argument('-ed','--end_date', type=str, default='0059-01-01',
                     help='''End year to compute averages. Default 0059-01-01''')
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
  case_config['Case'] = {'CASEROOT': args.caseroot,
                         'CIMEROOT': args.cimeroot}

  # Create the case instance
  dcase = DiagsCase(case_config['Case'])
  RUNDIR = dcase.get_value('RUNDIR')
  if args.debug:
    print('Run directory is:', RUNDIR)
    print('Casename is:', dcase.casename)
  # Update dict
  case_config['Case'].update({'RUNDIR' : RUNDIR})
  if not os.path.isdir(dcase.casename):
    print('Creating {}... \n'.format(dcase.casename))
    os.system('mkdir -p {}'.format(dcase.casename))
    make_yaml(dcase.casename, case_config)
    make_run_script(dcase.casename)
  else:
    print('Directory {} already exists. \n'.format(dcase.casename))

  return

def make_run_script(casename):
  """
  Create an ascii file to run a set of diagnostics.
  """
  print('Writing run_mom6_tools.sh...')
  f = open(casename+'/run_mom6_tools.sh', 'w')
  f.write("moc.py diag_config.yml -nw 6 &\n")
  f.write("poleward_heat_transport.py diag_config.yml -nw 6 &\n")
  f.write("section_transports.py diag_config.yml -save_ncfile &\n")
  f.write("surface.py diag_config.yml -nw 6 &\n")
  f.write("equatorial_comparison.py diag_config.yml -nw 6 &\n")
  f.write("stats.py diag_config.yml -diff_rms -nw 6 &\n")
  f.write("TS_levels.py diag_config.yml -nw 6 &\n")
  f.close()
  os.system('chmod +x '+casename+'/run_mom6_tools.sh')
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
