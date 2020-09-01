#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

class CutApplier:
    sf_list = list()
    cut_list = list()
    var_list = list()
    der_var_list = list()
    def __init__(self, arrays):
        self.arrays = arrays
        self.cuts = ak.Array([True]*len(arrays))
        self.all_vars = set()

        for cut_name in CutApplier.cut_list:
            full_cut_name = cut_name.replace("Event", "arrays.Event")
            self.cuts = eval("self.cuts and {}".format(full_cut_name))

        self.output = dict()
        # self.output = {"scale_factor": ak.Array([1.]*len(arrays[self.cuts]))}
        # for scale_name in CutApplier.sf_list:
        #     self.output["scale_factor"] = (self.output["scale_factor"]
        #                                    * arrays[scale_name][self.cuts])
        for group, add_vars, _ in CutApplier.var_list:
            self.output[group] = ak.Array({})
            self.all_vars |= set(add_vars)
            
        for group, add_vars in CutApplier.der_var_list:
            self.output[group] = ak.Array({})
            for var in add_vars:
                self.output[group][var] = self.arrays[var][self.cuts]

    @staticmethod
    def add_scale_factor(scale_name):
        CutApplier.sf_list.append(scale_name)

    @staticmethod
    def add_cut(cut_name):
        CutApplier.cut_list.append(cut_name)

    @staticmethod
    def add_vars(groupName, var_list, mask=None):
        CutApplier.var_list.append((groupName, var_list, mask))

    @staticmethod
    def add_vars_derived(groupName, var_list):
        CutApplier.der_var_list.append((groupName, var_list))

    def run(self, filename):
        start, end = 0, 0
        for array in uproot.iterate("{}:Events".format(filename), self.all_vars):
            end += len(array)
            mask = self.cuts[start:end]
            for group, add_vars, mask_name in CutApplier.var_list:
                if mask_name is not None:
                    submask = self.arrays[mask_name][start:end]
                    subarray = array[add_vars][submask]
                    subarray = subarray[mask]
                else:
                    subarray = array[mask]
                
                if len(self.output[group]) == 0:
                    self.output[group] = subarray
                else:
                    self.output[group] = ak.concatenate([self.output[group], subarray])

            start = end
