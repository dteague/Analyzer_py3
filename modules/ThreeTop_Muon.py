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
        loose = self.prefix(head, ["pt", 'dxy', 'dz',  "isGlobal", "isTracker",
                                    "isPFcand", 'miniPFRelIso_all',])
        tight = self.prefix(head, ["pt", 'dxy', 'dz',  "isGlobal", "isTracker",
                           "isPFcand", 'miniPFRelIso_all',])
        self.variables = np.unique(np.concatenate((loose, tight)))

        self.extraFuncs = [("loose_mask", "Muon_looseMask", None, loose)]




    # def cut(self, muons):
    #     return muons[muons.Muon_pt > 5 and
    #                     (muons.Muon_isGlobal or muons.Muon_isTracker) and
    #                     muons.Muon_isPFcand and
    #                     muons.Muon_miniPFRelIso_all < 0.4 and
    #                     np.abs(muons.Muon_dz) < 0.1 and
    #                     np.abs(muons.Muon_dxy) < 0.05]

    # Numba methods
    
    @staticmethod
    @numba.jit(nopython=True)
    def loose_mask(events, builder):
        for event in events:
            builder.begin_list()
            for j in range(len(event["Muon_pt"])):
                builder.boolean(
                    event.Muon_pt[j] > 5 and
                    # (event.Muon_isGlobal[j] or event.Muon_isTracker[j]) and
                    #     event.Muon_isPFcand[j] and
                    event.Muon_miniPFRelIso_all[j] < 0.4 and
                    np.abs(event.Muon_dz[j]) < 0.1 and
                    np.abs(event.Muon_dxy[j]) < 0.05
                )
            builder.end_list()



mu = Muon()

mu.run()
