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
                     inmask="Jet_rmCloseJet",
                     vals = Jet.jet)
        self.add_job("bjet_mask", outmask = "Jet_bjetMask",
                     inmask="Jet_rmCloseJet",
                     vals = Jet.bjet)

    # Numba methods
    close_jet = ["Muon_eta", "Muon_phi", "Jet_eta", "Jet_phi",
                 "Electron_eta", "Electron_phi"]
    @staticmethod
    @numba.jit(nopython=True)
    def closeJet(events, builder):
        j = 0
        for event in events:
            builder.begin_list()
            close_jet = []
            for i in range(len(event.Electron_closeJetIndex)):
                if j == 3:
                    print(event.Electron_closeJetIndex[i][0], event.Electron_closeJetIndex[i][1])
                if event.Electron_closeJetIndex[i][1] < 0.16:
                    close_jet.append(int(event.Electron_closeJetIndex[i][0]))
            for i in range(len(event.Muon_closeJetIndex)):
                if event.Muon_closeJetIndex[i][1] < 0.16:
                    close_jet.append(int(event.Muon_closeJetIndex[i][0]))
            # print(close_jet)
            for jidx in range(len(event.Jet_eta)):
                isClose = jidx in close_jet
                builder.boolean(not isClose)
            builder.end_list()
            j+=1


    jet = pre("Jet", ["pt", "eta", "jetId"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,i4)')
    def jet_mask(pt, eta, jetId):
        jetId_key = 0b11
        return (
            pt > 40 and
            abs(eta) < 2.4 and
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
        return (True
            # pt > 25 and
            # abs(eta) < 2.4 and
            # (jetId & jetId_key) != 0 and
            # btag > btag_cut
        )
