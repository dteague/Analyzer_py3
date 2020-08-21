#!/usr/bin/env python3

import python.Process as Process
import awkward1 as ak
import numpy as np

class Scheduler:
    jobs = list()
    def __init__(self, group, files):

        self.masks = Process()
        self.group = group
        self.files = files

    @staticmethod
    def add_step(class_list):
        Scheduler.jobs.append(class_list)

    def run(self):
        self.masks = Process()
        for job in Scheduler.jobs:
            classes = [cls(self.masks) for cls in job]
            for cls in classes:
                cls.run(self.files)
            for process in classes:
                self.masks += process
                
    def add_tree(self):
        total_mask = ak.Array({})
        for key, arr in self.masks.outmasks.items():
            print(key)
            total_mask[key] = arr
        print(total_mask[:10].tolist())
        # ak.to_parquet(total_mask, "test/{}.parquet".format(self.group))
