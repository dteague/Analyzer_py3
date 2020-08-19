#!/usr/bin/env python3

from python.Process import Process

class Job:
    def __init__(self, classes):
        self.classes = classes
        
    def run(self, filename, pool=None):
        classes = [cls() for cls in self.classes]
        if pool is not None:
            [pool.apply_async(cls.run, (filename)) for cls in classes]
            pool.join()
        else:
            [cls.run(filename) for cls in classes]

        total_process = Process()
        for process in classes:
            total_process += process

        return total_process
