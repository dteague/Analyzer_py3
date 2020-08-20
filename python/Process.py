#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

class Process:
    def __init__(self, process = None):
        self.extraFuncs = list()
        if process is None:
            self.masks = list()
        else:
            self.masks = process.masks
                    
    def __iadd__(self, other):
        if not self.masks:
            self.masks = [ak.Array({}) for i in range(len(other.masks))]
        for i in range(len(other.masks)):
            for col in other.masks[i].columns:
                self.masks[i][col] = other.masks[i][col]
        return self

    def run(self, filename):
        allvars = self.get_all_vars()
        for i, array in enumerate(uproot.iterate("{}:Events".format(filename), allvars)):
            for func, write_name, org_mask, var in self.extraFuncs:
                self.masks.append(ak.Array({}))
                events = array[var]
                if isinstance(org_mask, str):
                    events = events[self.masks[i][org_mask]]
                elif isinstance(org_mask, dict):
                    for val, mask_name in org_mask.items():
                        var_apply = np.array(events.columns)[[val in col for col in events.columns]]
                        for col in var_apply:
                            events[col] = events[col][self.masks[i][mask_name]]
                # For different runtypes
                #print(func, repr(getattr(self, func)))
                if self.isJit(func):
                    mask = ak.ArrayBuilder()
                    getattr(self, func)(events,  mask)
                    self.masks[i][write_name] = mask.snapshot()
                elif self.isVectorize(func):
                    variables = [events[col] for col in var]
                    #print([ak.type(v[0]) for v in variables])
                    self.masks[i][write_name] = getattr(self, func)(*variables)
                else:
                    mask = ak.ArrayBuilder()
                    self.masks[i][write_name] = getattr(self, func)(events[var])

    def get_all_vars(self):
        return_set = set()
        for _, _, _, var_list in self.extraFuncs:
            return_set |= set(var_list)
        return list(return_set)

    @staticmethod
    def prefix(pre, lister):
        return ["_".join([pre, l]) for l in lister]
    
    def isJit(self, funcName):
        return "Dispatcher" in repr(getattr(self, funcName))

    def isVectorize(self, funcName):
        return "DUFunc" in repr(getattr(self, funcName))
