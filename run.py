#!/usr/bin/env python3

from python.Scheduler import Scheduler
from modules import Electron, Muon, Jet, EventWide
from threading import Thread
#mp.set_start_method("spawn")




if __name__ == "__main__":
    files_dict = {"TTT1": "tree_1.root", "TTT2": "tree_2.root",
             "TTT3": "tree_3.root"}
    Scheduler.add_step([Muon, Electron])
    Scheduler.add_step([Jet])
    Scheduler.add_step([EventWide])


    jobs = list()
    for group, files in files_dict.items():
        print(group)
        schedule = Scheduler(group, files)
        p = Thread(target=schedule.run)
        jobs.append(p)
        p.start()
