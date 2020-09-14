#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba
from anytree import Node


class Process:
    def __init__(self, process = None):
        self.extraFuncs = list()
        if process is None:
            self.outmasks = dict()
            self.mask_tree = dict()
        else:
            self.outmasks = process.outmasks
            self.mask_tree = process.mask_tree

    def __iadd__(self, other):
        if isinstance(other, Process):
            for col in other.outmasks:
                self.outmasks[col] = other.outmasks[col]
            self.mask_tree.update(other.mask_tree)
        else:
            for col in other.columns:
                self.outmasks[col] = other[col]
        
        return self

    def add_job(self, func, outmask, vals, inmask=None, addvals={}):
        inmask_dict = dict()
        if isinstance(inmask, list):
            for mask in inmask:
                key = mask.split("_")[0] + "_"
                var_apply = [col for col in vals if key in col]
                inmask_dict[mask] = var_apply
                self.mask_tree[outmask] = Node(outmask, self.mask_tree[mask])
        elif isinstance(inmask, str):
            key = inmask.split("_")[0] + "_"
            var_apply = [col for col in vals if key in col]
            inmask_dict[inmask] = var_apply
            self.mask_tree[outmask] = Node(outmask, self.mask_tree[inmask])
        elif inmask is None:
            self.mask_tree[outmask] = Node(outmask, Node("base"))
        self.outmasks[outmask] = ak.Array([])

        self.extraFuncs.append((func, outmask, inmask_dict, vals, addvals))

    def run(self, filename):
        allvars = self.get_all_vars()
        start, end = 0, 0
        for array in uproot.iterate("{}:Events".format(filename), allvars):
            end += len(array)
            print("Events considered: ", end)
            for func, write_name, inmask, var, addvals in self.extraFuncs:
                events = array[var]
                for mask_name, vals in inmask.items():
                    submasks = self.get_masks(mask_name, start, end)
                    for submask in submasks:
                        for col in vals:
                            events[col] = events[col][self.outmasks[submask][start:end]]
                        
                for addval, mask in addvals.items():
                    events[addval] = self.add_var(mask, addval, start, end)

                # For different runtypes
                final_mask = None
                if self.isJit(func):
                    mask = ak.ArrayBuilder()
                    getattr(self, func)(events, mask)
                    final_mask = mask.snapshot()
                elif self.isVectorize(func):
                    variables = [events[col] for col in var+list(addvals.keys())]
                    # print([ak.type(v[0]) for v in variables])
                    final_mask = getattr(self, func)(*variables)
                else:
                    final_mask = getattr(self, func)(events[var])
                
                self.outmasks[write_name] = ak.concatenate(
                    [self.outmasks[write_name], final_mask])
                                
            start = end
            
    def get_all_vars(self):
        return_set = set()
        for _, _, _, var_list, _ in self.extraFuncs:
            return_set |= set(var_list)
        return list(return_set)
    
    def isJit(self, funcName):
        return "Dispatcher" in repr(getattr(self, funcName))

    def isVectorize(self, funcName):
        return "DUFunc" in repr(getattr(self, funcName))

    def get_masks(self, mask_name, start, end):
        apply_list = list()
        node = self.mask_tree[mask_name]
        while node.name != "base":
            apply_list.append(node.name)
            node = node.parent
        return apply_list[::-1]
        # total_mask = self.outmasks[apply_list.pop()][start:end]

        # for m_name in apply_list[::-1]:
        #     total_mask = total_mask[self.outmasks[m_name][start:end]]
        #     print(self.outmasks[m_name][start:start+4].tolist())

        # return total_mask

    def add_var(self, mask_name, var_name, start, end):
        variable = self.outmasks[var_name][start:end]
        if mask_name is None:
            return variable
        var_parent = self.mask_tree[var_name].parent.name
        work_node = self.mask_tree[mask_name]
        apply_list = list()
        
        while work_node.name != var_parent:
            apply_list.append(work_node.name)
            work_node = work_node.parent
            
        for m_name in apply_list[::-1]:
            variable = variable[self.outmasks[m_name][start:end]]
        return variable
