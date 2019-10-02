#import esmlab
import xarray as xr
import logging
import os
import re
import cftime as cft
import logging as log
from DiagsCase import DiagsCase
from esmlab import climatology

class ClimoGenerator(object,):

    def __init__(self, cid, climo_config:dict, diags_case:DiagsCase):
        self._cid = cid
        self.climo_config = climo_config
        self.diags_case = diags_case

        self._fields = climo_config['fields']

        date_regex = "^\d{4}-\d{2}-\d{2}$"
        self._date0 = None
        self._date1 = None
        if 'date0' in climo_config:
            date0_str = climo_config['date0']
            assert re.search(date_regex,date0_str), f"Invalid date0 entry: {date0_str}"
            self._date0 = cft.datetime(int(date0_str[:4]),int(date0_str[5:7]),int(date0_str[8:10]))
            log.info(f"Climo {self._cid}, date0: {self._date0}")
        if 'date1' in climo_config:
            date1_str = climo_config['date1']
            assert re.search(date_regex,date1_str), f"Invalid date1 entry: {date1_str}"
            self._date1 = cft.datetime(int(date1_str[:4]),int(date1_str[5:7]),int(date1_str[8:10]))
            log.info(f"Climo {self._cid}, date1: {self._date1}")



    @property
    def fields(self):
        if len(self._fields)==0:
            raise RuntimeError("fields not initialized")
        return self._fields

    @property
    def date0(self):
        return self._date0

    @property
    def date1(self):
        return self._date1
