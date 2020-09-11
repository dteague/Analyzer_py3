#!/usr/bin/env python3

from python.Scheduler import Scheduler
from python.CutApplier import CutApplier
import modules
from threading import Thread
from queue import Queue
import Utilities.FileGetter as fg
from Utilities.FileGetter import pre

import warnings
warnings.filterwarnings('ignore')

def job_run(job_type, *args):
    job = Scheduler(*args)
    if job_type == "create":
        job.run()
        job.add_tree()
    elif job_type == "apply":
        job.apply_mask()
    elif job_type == "all":
        job.run()
        job.add_tree()
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


