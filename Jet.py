#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from segment import Process

class Jet(Process):
    def __init__(self):
        super().__init__()

        head = "Jet"
        jet = self.prefix(head, ["pt", "eta", "jetId"])
        bjet = self.prefix(head, ["pt", "eta", "jetId", "btagDeepB"])
        ht = ["Jet_pt"]

        self.extraFuncs = [("jet_mask", "Jet_jetMask", None, jet),
                           ("bjet_mask", "Jet_bjetMask", None, bjet),
                           ("calc_HT", "Jet_HT", "Jet_jetMask", ht)]
        
    # Numba methods
    @staticmethod
    @numba.jit(nopython=True)
    def jet_mask(events, builder):
        jetId = 0b11
        for event in events:
            builder.begin_list()
            for j in range(len(event["Jet_pt"])):
                builder.boolean(
                    event.Jet_pt[j] > 40 and
                    np.abs(event.Jet_eta[j]) < 2.4 and
                    (event.Jet_jetId[j] & jetId) != 0
                )
            builder.end_list()

    @staticmethod
    @numba.jit(nopython=True)
    def bjet_mask(events, builder):
        jetId = 0b11
        for event in events:
            builder.begin_list()
            for j in range(len(event["Jet_pt"])):
                builder.boolean(
                    event.Jet_pt[j] > 25 and
                    np.abs(event.Jet_eta[j]) < 2.4 and
                    event.Jet_btagDeepB[j] > 0.6324 and
                    (event.Jet_jetId[j] & jetId) != 0
                )
            builder.end_list()

    @staticmethod
    @numba.jit(nopython=True)
    def calc_HT(events, builder):
        for event in events:
            HT = 0
            for j in range(len(event["Jet_pt"])):
                HT += event.Jet_pt[j]
            builder.real(HT)
