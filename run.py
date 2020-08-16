#!/usr/bin/env python3

from python.Scheduler import Scheduler

from modules import Electron, Muon, Jet


schedule = (
    ( Electron() | Muon() ) &
    Jet()
)
schedule.run()
