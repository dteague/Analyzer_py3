#!/usr/bin/env python3

from python.Scheduler import Scheduler
from python.CutApplier import CutApplier
from modules import Electron, Muon, Jet, EventWide
from threading import Thread
from queue import Queue
import Utilities.FileGetter as fg
import warnings
warnings.filterwarnings('ignore')

# Run Specifics
Scheduler.add_step([Muon, Electron])
Scheduler.add_step([Jet])
Scheduler.add_step([EventWide])

# Applying specifics
CutApplier.add_scale_factor("Event_wDecayScale")
CutApplier.add_scale_factor("Event_pileupScale")
CutApplier.add_cut("Event_MetFilterMask")
CutApplier.add_cut("Event_triggerMask")
CutApplier.add_cut("abs(Event_channels) > 1")
CutApplier.add_vars(["Muon_pt"])


def job_run(job_type, *args):
    job = Scheduler(*args)
    if job_type == "create":
        job.run()
        job.add_tree()
    elif job_type == "apply":
        job.apply_mask()
    else:
        print("problem with job")


def worker():
    while True:
        job_type, group, files, outdir = q.get()
        job_run(job_type, group, files, outdir)
        q.task_done()


if __name__ == "__main__":
    args = fg.get_generic_args()

    info = fg.FileGetter(args.analysis, args.selection)
    files_dict = info.get_file_dict(args.filenames)
    fg.checkOrCreateDir(args.outdir)

    argList = list()
    for group, files in files_dict.items():
        argList.append((args.proc_type, group, files, args.outdir))

    if args.j == 1:
        for arg in argList:
            job_run(arg[0], *arg[1:])
    else:
        q = Queue()
        for _ in range(args.j):
            t = Thread(target=worker)
            t.daemon = True
            t.start()
        for arg in argList:
            q.put(arg)

        q.join()       # block until all tasks are done


