#!/usr/bin/env python3

from python.Scheduler import Scheduler
from modules import Electron, Muon, Jet, EventWide
from threading import Thread
import Utilities.FileGetter as fg
import warnings
warnings.filterwarnings('ignore')

Scheduler.add_step([Muon, Electron])
Scheduler.add_step([Jet])
Scheduler.add_step([EventWide])

if __name__ == "__main__":
    info = fg.FileGetter("ThreeLep", "TwoLep_Met25")

    #files_dict = info.get_file_dict(["xg"])
    files_dict = info.get_file_dict("ttg_lepfromTbar")

    jobs = list()

    for group, files in files_dict.items():
        print(group)
        schedule = Scheduler(group, files)
        schedule.run()
        jobs.append((0, schedule))

        # job = Thread(target=schedule.run)
        # jobs.append((job, schedule))
        # job.start()

    # for job, _ in jobs:
    #     job.join()
    
    for _, sched in jobs:
            sched.add_tree()
