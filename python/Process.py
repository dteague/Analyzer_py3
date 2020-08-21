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
        else:
            self.outmasks = process.outmasks
        self.mask_tree_base = Node("base")
        self.mask_tree = dict()

    def __iadd__(self, other):
        for col in other.outmasks:
            self.outmasks[col] = other.outmasks[col]
        return self

    def add_job(self, func, outmask, vals, inmask=None, addvals=None):
        inmask_dict = dict()
        if isinstance(inmask, dict):
            for mask, key in inmask.items():
                var_apply = [col for col in vals if key in col]
                inmask_dict[mask] = var_apply
                self.mask_tree[outmask] = Node(outmask, self.mask_tree[mask])
        elif isinstance(inmask, str):
            inmask_dict[inmask] = vals
            self.mask_tree[outmask] = Node(outmask, self.mask_tree[inmask])
        elif inmask is None:
            self.mask_tree[outmask] = Node(outmask, self.mask_tree_base)
        self.outmasks[outmask] = ak.Array([])

        self.extraFuncs.append((func, outmask, inmask_dict, vals, addvals))

    def run(self, filename):
        allvars = self.get_all_vars()
        start = 0
        for array in uproot.iterate("{}:Events".format(filename), allvars):
            for func, write_name, inmask, var, addvals in self.extraFuncs:
                events = array[var]
                for mask, vals in inmask.items():
                    print(mask)
                    for col in vals:
                        events[col] = events[col][self.get_mask(mask, start)]

                if isinstance(addvals, dict):
                    for addval, mask in addvals.items():
                        # print(self.add_var(mask, addval, start)[:10].tolist())
                        # print(events[:10].tolist())
                        events[addval] = self.add_var(mask, addval, start)

                # For different runtypes
                final_mask = None
                if self.isJit(func):
                    mask = ak.ArrayBuilder()
                    getattr(self, func)(events,  mask)
                    final_mask = mask.snapshot()
                    final_mask[:5]
                elif self.isVectorize(func):
                    variables = [events[col] for col in var]
                    #print([ak.type(v[0]) for v in variables])
                    final_mask = getattr(self, func)(*variables)
                else:
                    final_mask = getattr(self, func)(events[var])
                
                self.outmasks[write_name] = ak.concatenate(
                    [self.outmasks[write_name], final_mask])
                                
            start += len(array)
            
    def get_all_vars(self):
        return_set = set()
        for _, _, _, var_list, _ in self.extraFuncs:
            return_set |= set(var_list)
        return list(return_set)

    @staticmethod
    def prefix(pre, lister):
        return ["_".join([pre, l]) for l in lister]
    
    def isJit(self, funcName):
        return "Dispatcher" in repr(getattr(self, funcName))

    def isVectorize(self, funcName):
        return "DUFunc" in repr(getattr(self, funcName))

    def get_mask(self, mask_name, start):
        apply_list = list()
        node = self.mask_tree[mask_name]
        while node.name != "base":
            apply_list.append(node.name)
            node = node.parent
        total_mask = self.outmasks[apply_list.pop()][start:]
        for m_name in apply_list[::-1]:
            total_mask = total_mask[self.outmasks[m_name][start:]]
        return total_mask

    def add_var(self, mask_name, var_name, start):
        print(self.mask_tree.items())
        variable = self.outmasks[var_name][start:]
        var_parent = self.mask_tree[var_name].parent.name
        work_node = self.mask_tree[mask_name]
        apply_list = list()
        
        while work_node.parent.name != var_parent:
            apply_list.append(work_node.name)
            work_node = work_node.parent

        for m_name in apply_list[::-1]:
            variable = variable[self.outmasks[m_name][start:]]
        return variable
