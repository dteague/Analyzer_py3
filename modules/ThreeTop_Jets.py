#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from python.Process import Process
from Utilities.FileGetter import pre

class Jet(Process):
    def __init__(self, process):
        super().__init__(process)

        self.add_job("closeJet", outmask="Jet_rmCloseJet", vals = Jet.close_jet,
                     addvals = {"Electron_closeJetIndex": "Electron_fakeMask",
                                "Muon_closeJetIndex": "Muon_fakeMask"})
        self.add_job("jet_mask", outmask = "Jet_jetMask",
                     inmask="Jet_rmCloseJet", vals = Jet.jet)
        self.add_job("bjet_mask", outmask = "Jet_bjetMask",
                     inmask="Jet_rmCloseJet", vals = Jet.bjet)

    # Numba methods
    close_jet = ["Muon_eta", "Muon_phi", "Jet_eta", "Jet_phi",
                 "Electron_eta", "Electron_phi"]
    @staticmethod
    @numba.jit(nopython=True)
    def closeJet(events, builder):
        for event in events:
            builder.begin_list()
            for jidx in range(len(event.Jet_eta)):
                dr = 10
                for eidx in range(len(event.Electron_closeJetIndex)):
                    if jidx == event.Electron_closeJetIndex[eidx]:
                        dr = (event.Electron_eta[eidx] - event.Jet_eta[jidx])**2 \
                            + (event.Electron_phi[eidx] - event.Jet_phi[jidx])**2
                        break
                if dr < 0.16:   # 0.4**2
                    builder.boolean(False)
                    continue
                for midx in range(len(event.Muon_closeJetIndex)):
                    if jidx == event.Muon_closeJetIndex[midx]:
                        dr = (event.Muon_eta[midx] - event.Jet_eta[jidx])**2 \
                            + (event.Muon_phi[midx] - event.Jet_phi[jidx])**2
                        break
                builder.boolean(dr > 0.16 )
            builder.end_list()



    jet = pre("Jet", ["pt", "eta", "jetId"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,i4)')
    def jet_mask(pt, eta, jetId):
        jetId_key = 0b11
        return (
            pt > 40 and
            np.abs(eta) < 2.4 and
            (jetId & jetId_key) != 0
        )
                                                            
    bjet = pre("Jet", ["pt", "eta", "jetId", "btagDeepB"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,i4,f4)')
    def bjet_mask(pt, eta, jetId, btag):
        jetId_key = 0b11
        btag_cut = 0.6324 # 2016
        # btag_cut = 0.4941 # 2017
        # btag_cut = 0.4184 # 2018
        return (
            pt > 25 and
            np.abs(eta) < 2.4 and
            (jetId & jetId_key) != 0 and
            btag > btag_cut
        )
