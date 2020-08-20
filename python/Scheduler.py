#!/usr/bin/env python3

import python.Process as Process


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
        total_process = Process()
        for job in Scheduler.jobs:
            classes = [cls(total_process) for cls in job]
            for cls in classes:
                cls.run(self.files)
            for process in classes:
                total_process += process
