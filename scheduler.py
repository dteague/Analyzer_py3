#!/usr/bin/env python3

class Scheduler():
    def __init__(self, proc=None):
        if proc is None:
            self.process_list = list()
        else:
            self.process_list = proc

    def __or__(self, other):
        if isinstance(other, Scheduler):
            return Scheduler(self.process_list + other.process_list)
        else:
            return Scheduler(self.process_list + [other])


    def __and__(self, other):
        if isinstance(other, Scheduler):
            return Scheduler([self.process_list, other.process_list])
        else:
            return Scheduler([self.process_list, [other]])

    def run(self):
        self.recursive_run(self.process_list)

    def recursive_run(self, run_list):
        for item in run_list:
            if isinstance(item, list):
                self.recursive_run(item)
            else:
                print(item)
                item.run()

    def __str__(self):
        return str(self.process_list)
