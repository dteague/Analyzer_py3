#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from python.Process import Process

class Muon(Process):
    def __init__(self):
        super().__init__()

        head = "Muon"
        loose = self.prefix(head, ["pt", 'dxy', 'dz',"isGlobal", "isTracker",
                                    "isPFcand", 'miniPFRelIso_all',])
        tight = self.prefix(head, ["pt", 'miniPFRelIso_all', "tightCharge",
                                   "mediumId", "sip3d"])
        
        self.extraFuncs = [
            ("loose_mask", "Muon_looseMask", None, loose),
            ("tight_mask", "Muon_tightMask", "Muon_looseMask", tight)
        ]

    # Numba methods
    
    @staticmethod
    @numba.jit(nopython=True)
    def loose_mask(events, builder):
        for event in events:
            builder.begin_list()
            for j in range(len(event["Muon_pt"])):
                builder.boolean(
                    event.Muon_pt[j] > 5 and
                    (event.Muon_isGlobal[j] == 0 or event.Muon_isTracker[j] == 0) and
                    event.Muon_isPFcand[j] == 1 and
                    event.Muon_miniPFRelIso_all[j] < 0.4 and
                    np.abs(event.Muon_dz[j]) < 0.1 and
                    np.abs(event.Muon_dxy[j]) < 0.05
                )
            builder.end_list()

    @staticmethod
    @numba.jit(nopython=True)
    def tight_mask(events, builder):
        for event in events:
            builder.begin_list()
            for j in range(len(event["Muon_pt"])):
                builder.boolean(
                    event.Muon_pt[j] > 20 and
                    event.Muon_miniPFRelIso_all[j] < 0.16 and
                    event.Muon_tightCharge[j] == 2 and
                    event.Muon_mediumId[j] == 1 and
                    event.Muon_sip3d[j] < 4
                )
            builder.end_list()

