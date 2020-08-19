#!/usr/bin/env python3

import python.Job as Job
import python.Process as Process

class Scheduler:
    def __init__(self):
        self.jobs = []
        self.mask_dict = dict()
        self.file_dict = dict()

    def add_step(self, class_list):
        self.jobs.append(Job(class_list))

    def set_files(self, file_dict):
        self.file_dict = file_dict
        for key, val in file_dict.items():
            self.mask_dict[key] = Process()

    def run(self):
        for group, files in self.file_dict.items():
            for job in self.jobs:
                self.mask_dict[group] += job.run(files)
