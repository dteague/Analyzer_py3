#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

class CutApplier:
    sf_list = list()
    cut_list = list()
    var_list = list()
    def __init__(self, arrays):
        self.arrays = arrays
        self.cuts = ak.Array([True]*len(self.arrays))
        self.all_vars = list()

        self.output = {"scale_factor": ak.Array([1.]*len(self.arrays))}
        for scale_name in CutApplier.sf_list:
            self.output["scale_factor"] = (self.output["scale_factor"]
                                           * self.arrays[scale_name])
        for cut_name in CutApplier.cut_list:
            full_cut_name = cut_name.replace("Event", "self.arrays.Event")
            self.cuts = eval("self.cuts and {}".format(full_cut_name))

    @staticmethod
    def add_scale_factor(scale_name):
        CutApplier.sf_list.append(scale_name)

    @staticmethod
    def add_cut(cut_name):
        CutApplier.cut_list.append(cut_name)

    @staticmethod
    def add_vars(var_list):
        CutApplier.var_list = var_list

    def run(self, filename):
        for var in CutApplier.var_list:
            self.output[var] = ak.Array([])

        start, end = 0, 0
        for array in uproot.iterate("{}:Events".format(filename), CutApplier.var_list):
            end += len(array)
            array = array[self.cuts[start:end]]
            for col in array.columns:
                self.output[col] = ak.concatenate(
                    [self.output[col], array[col]])
            start = end
        print(len(self.output["Muon_pt"]))
