#!/usr/bin/env python3

from scheduler import Scheduler

from Electron import Electron
from Muon import Muon
from Jet import Jet


schedule = (Electron() | Muon()) & Jet()
schedule.run()
