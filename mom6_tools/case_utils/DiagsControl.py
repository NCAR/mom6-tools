import yaml
import os, sys
import logging as log
from collections import namedtuple
from DiagsCase import DiagsCase
from ClimoGenerator import ClimoGenerator

class DiagsControl(object,):

    def __init__(self, diag_config_yml_path):

        # load diag_config.yml file:
        self._config = yaml.load(open(diag_config_yml_path,'r'), Loader=yaml.Loader)

        # instantiate DiagsCase:
        self._case = DiagsCase(self._config['Case'])

    @property
    def case(self):
        return self._case


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    config_yml_path = "/glade/work/altuntas/mom6.diags/g.c2b6.GJRA.TL319_t061.long_JRA_mct.001/diag_config.yml"

    diags_control = DiagsControl(config_yml_path)
    diags_case = diags_control.case
    climo_gen = ClimoGenerator(diags_case)
    #climo_gen._construct_dataset(["temp", "salt"])    
    #computed_dset = climatology(climo_gen.dset, freq='mon')
    #computed_dset.to_netcdf("/glade/scratch/altuntas/climo.nc")

