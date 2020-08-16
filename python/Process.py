#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from python.Scheduler import Scheduler

class Process:
    def __init__(self, array=None):
        self.extraFuncs = list()
        if array is None:
            self.masks = ak.Array({})
        else:
            self.masks = array

    def __add__(self, other):
        proc = Process(self.masks)
        for col in other.masks.columns:
            proc.masks[col] = other.masks[col]
        return proc

    def __iadd__(self, other):
        for col in other.masks.columns:
            self.masks[col] = other.masks[col]

    def __or__(self, other):
        if isinstance(other, Scheduler):
            return Scheduler([self] + other.process_list)
        else:
            return Scheduler([self] + [other])

    def __and__(self, other):
        if isinstance(other, Scheduler):
            return Scheduler([[self], other.process_list])
        else:
            return Scheduler([[self], [other]])

    def run(self):
        with uproot.open("tree_1.root") as file:
            for func, write_name, org_mask, var in self.extraFuncs:
                mask = ak.ArrayBuilder()
                events = file["Events"].arrays(var)
                if org_mask is not None:
                    events = events[self.masks[org_mask]]
                for col in events.columns:
                    if "bool" in repr(ak.type(events[col])):
                        events[col] = events[col] + 0
                getattr(self, func)(events[var],  mask)
                self.masks[write_name] = mask.snapshot()

    @staticmethod
    def prefix(pre, lister):
        return ["_".join([pre, l]) for l in lister]
    
