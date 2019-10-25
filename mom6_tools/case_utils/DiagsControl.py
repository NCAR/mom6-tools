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
        try:
            self._case = DiagsCase(self._config['Case'])
        except KeyError:
            raise KeyError('Must provide a "Case" entry in diag_config file')

        try:
            self._climo_entries = self._config['Climatology']
        except KeyError:
            raise KeyError('Must provide a "Climatology" entry in diag_config file')

    @property
    def case(self):
        return self._case

    @property
    def climo_entries(self):
        return self._climo_entries

if __name__ == '__main__':

    log.basicConfig(level=log.INFO)
    config_yml_path = "/glade/work/altuntas/mom6.diags/g.c2b6.GJRA.TL319_t061.long_JRA_mct.001/diag_config.yml"
    dc = DiagsControl(config_yml_path)

    for climo_id in dc.climo_entries:
        log.info(f"Creating climo {climo_id}")
        climo = ClimoGenerator(climo_id, dc.climo_entries[climo_id], dc.case)
        field_list = climo.fields
        esm = dc.case.create_dataset(field_list)
        print(type(esm.time))

