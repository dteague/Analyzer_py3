#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from python.Process import Process

class Electron(Process):
    def __init__(self):
        super().__init__()

        head = "Electron"

        loose = self.prefix(head, ["pt", "convVeto", "lostHits", "dz", "dxy",
                                   "miniPFRelIso_all",])
        emu = self.prefix(head, ["eta", "sieie", "hoe", "eInvMinusPInv"])
        tight = self.prefix(head, ["pt", "lostHits", "tightCharge", "sip3d",
                                   "miniPFRelIso_all", "dr03EcalRecHitSumEt",
                                   "dr03HcalDepth1TowerSumEt", "dr03TkSumPt"])
        
        self.extraFuncs = [
            ("loose_mask", "Electron_looseMask", None, loose),
            ("trigger_emu", "Electron_looseMask", "Electron_looseMask", emu),
            ("tight_mask", "Electron_tightMask", "Electron_looseMask", tight),
        ]
        
    # # Numba methods
    @staticmethod
    @numba.jit(nopython=True)
    def loose_mask(events, builder):
        for event in events:
            builder.begin_list()
            for j in range(len(event["Electron_pt"])):
                builder.boolean(
                    event.Electron_pt[j] > 7 and
                    event.Electron_convVeto[j] == 1 and
                    event.Electron_lostHits[j] <= 1 and
                    event.Electron_miniPFRelIso_all[j] < 0.4 and
                    np.abs(event.Electron_dz[j]) < 0.1 and
                    np.abs(event.Electron_dxy[j]) < 0.05
                )
            builder.end_list()

    @staticmethod
    @numba.jit(nopython=True)
    def tight_mask(events, builder):
        for event in events:
            builder.begin_list()
            for j in range(len(event["Electron_pt"])):
                pt = event.Electron_pt[j]
                builder.boolean(
                    pt > 20 and
                    event.Electron_lostHits[j] == 0 and
                    event.Electron_tightCharge[j] == 2 and
                    event.Electron_miniPFRelIso_all[j] < 0.12 and
                    event.Electron_sip3d[j] < 4 and
                    event.Electron_dr03EcalRecHitSumEt[j] / pt < 0.45 and
                    event.Electron_dr03HcalDepth1TowerSumEt[j] / pt < 0.25 and
                    event.Electron_dr03TkSumPt[j] / pt < 0.2
                )
            builder.end_list()


    @staticmethod
    @numba.jit(nopython=True)
    def trigger_emu(events, builder):
        for event in events:
            builder.begin_list()
            for j in range(len(event["Electron_hoe"])):
                passed = ((event.Electron_eInvMinusPInv[j]) < 0.01 and
                          event.Electron_hoe[j] < 0.08)
                if not passed:
                    builder.boolean(False)
                    continue
                if abs(event.Electron_eta[j]) < 1.479:
                    builder.boolean(event.Electron_sieie[j] < 0.011)
                else:
                    builder.boolean(event.Electron_sieie[j] < 0.031)
            builder.end_list()
