#!/usr/bin/env python3

from python.Scheduler import Scheduler
from python.CutApplier import CutApplier
from modules import set_channel
from threading import Thread
from queue import Queue
import Utilities.FileGetter as fg
from Utilities.FileGetter import pre

import warnings
import os
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
        job_type, group, files, outdir, xsec = q.get()
        job_run(job_type, group, files, outdir, xsec)
        q.task_done()


if __name__ == "__main__":
    args = fg.get_generic_args()

    if args.channel:
        set_channel(args.channel)
    info = fg.FileGetter(args.analysis, args.selection)
    files_dict = info.get_file_dict(args.filenames)
    fg.checkOrCreateDir(args.outdir)

    argList = list()
    for group, files in files_dict.items():
        mask_exists = os.path.isfile("{}/{}.parquet".format(args.outdir, group))
        if args.proc_type == "apply" and not mask_exists:
            print("Mask file doesn't exist, please create!")
            exit(1)
        elif args.proc_type == "create" and not args.r and mask_exists:
            continue
        argList.append((args.proc_type, group, files, args.outdir,
                        info.get_xsec(group)))

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


