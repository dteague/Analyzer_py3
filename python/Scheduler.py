#!/usr/bin/env python3

import python.Process as Process
import awkward1 as ak
import numpy as np

class Scheduler:
    jobs = list()
    write_list = list()
    def __init__(self, group, files, out_dir):

        self.process = Process()
        self.group = group
        self.files = files
        self.out_dir = out_dir

    @staticmethod
    def add_step(class_list):
        Scheduler.jobs.append(class_list)



    def run(self):
        print("{}: Starting Job".format(self.group))
        for job in Scheduler.jobs:
            classes = [cls(self.process) for cls in job]
            for cls in classes:
                cls.run(self.files)
                self.process += cls
        print("{}: Finished Job".format(self.group))

        
    def add_tree(self):
        print("{}: Starting Write".format(self.group))
        total_mask = ak.Array({})
        for key, arr in self.process.outmasks.items():
            total_mask[key] = arr
        ak.to_parquet(total_mask, "{}/{}.parquet".format(self.out_dir, self.group))
        print("{}: Finished Write".format(self.group))

    def apply_mask(self):
        self.process += ak.from_parquet("{}/{}.parquet".format(self.out_dir, self.group))

        "Event_MetFilterMask"
        "Event_pileupScale"
        "Event_wDecayScale"
        "Event_triggerMask"
        "Event_channels"
