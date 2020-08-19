#!/usr/bin/env python3

from python.Scheduler import Scheduler
from modules import Electron, Muon, Jet, EventWide

     
schedule = Scheduler()
schedule.set_files({"TTT": "tree*.root"})

schedule.add_step([Muon, Electron])
schedule.add_step([Jet])
schedule.add_step([EventWide])

schedule.run()
