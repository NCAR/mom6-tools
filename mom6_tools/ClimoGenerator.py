import xarray as xr
import logging
import os
import re
import cftime as cft
import logging as log
from DiagsCase import DiagsCase
###from esmlab import climatology

class ClimoGenerator(object,):

    required_entries = set(["type", "date0", "date1", "fields", "freqs"])

    def __init__(self, cid, climo_config:dict, diags_case:DiagsCase):
        self._cid = cid
        self._climo_config = climo_config
        self.diags_case = diags_case

        # check if all the required entries are provided in climo_config:
        if len(self.required_entries - set(self._climo_config))>0:
            raise RuntimeError("Must provide all reqired fields:"+str(self.required_entries))

        self._fields = self._climo_config['fields']

        date_regex = "^\d{4}-\d{2}-\d{2}$"
        self._date0 = None
        self._date1 = None

        cft_datetime_constr= None
        calendar = diags_case.get_value("CALENDAR")
        if calendar == "NO_LEAP":
            cft_datetime_constr = cft.DatetimeNoLeap
        elif calendar == "GREGORIAN":
            cft_datetime_constr = cft.DatetimeGregorian
        else:
            raise RuntimeError(f"Unknown calendar type: {calendar}")

        allowable_types = ['avg','climo']
        if self._climo_config['type'] not in allowable_types:
            raise RuntimeError("The type must be set to one of: "+str(allowable_types))

        if 'date0' in self._climo_config:
            date0_str = self._climo_config['date0']
            assert re.search(date_regex,date0_str), f"Invalid date0 entry: {date0_str}"
            calendar = diags_case.get_value("CALENDAR")
            self._date0 = cft_datetime_constr(int(date0_str[:4]),int(date0_str[5:7]),int(date0_str[8:10]))
            log.info(f"Climo {self._cid}, date0: {self._date0}")
        if 'date1' in self._climo_config:
            date1_str = self._climo_config['date1']
            assert re.search(date_regex,date1_str), f"Invalid date1 entry: {date1_str}"
            calendar = diags_case.get_value("CALENDAR")
            self._date1 = cft_datetime_constr(int(date1_str[:4]),int(date1_str[5:7]),int(date1_str[8:10]))
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

    @property
    def type(self):
        return self._climo_config['type']

    @property
    def freqs(self):
        assert type(self._climo_config['freqs'])==list
        return self._climo_config['freqs']

    def stage(self):
        """ Creates datasets of requested climatologies/averages."""

        # first, stage the initial dataset covering the entire duration of the case
        case_dset = self.diags_case.stage_dset(self.fields)

        ## confine within climatology time frame
        date0_avail = case_dset.time.sel(time=self.date0,method="bfill")
        date1_avail = case_dset.time.sel(time=self.date1,method="pad")
        case_dset = case_dset.sel(time=slice(date0_avail,date1_avail))

        dset = dict() # class member that holds climatology dsets

        if self.type == 'avg':
            for freq in self.freqs:
                dset[freq] = case_dset.resample(time=freq, closed='left').mean()
        else:
            raise NotImplementedError

        return dset

