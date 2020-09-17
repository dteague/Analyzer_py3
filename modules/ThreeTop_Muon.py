#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba
import math

from python.Process import Process
from Utilities.FileGetter import pre
from python.Common import deltaR, jetRel, in_zmass

class Muon(Process):
    def __init__(self, process):
        super().__init__(process)

        self.add_job("loose_mask", outmask = "Muon_looseMask",
                     vals = Muon.loose)

        self.add_job("looseIdx", outmask = "Muon_looseIndex",
                     inmask = "Muon_looseMask", vals = ["Muon_pt"])

        self.add_job("fake_mask", outmask = "Muon_basicFakeMask",
                     inmask = "Muon_looseMask", vals = Muon.fake)
        self.add_job("closeJet", outmask = "Muon_closeJetIndex",
                     inmask = "Muon_basicFakeMask", vals = Muon.close_jet)
        self.add_job("fullIso", outmask = "Muon_fakeMask",
                     inmask = "Muon_basicFakeMask", vals = Muon.v_fullIso,
                     addvals = [(None, "Muon_closeJetIndex")])
        
        self.add_job("tight_mask", outmask = "Muon_finalMask",
                     inmask = "Muon_fakeMask", vals = Muon.tight)

        self.add_job("pass_zveto", outmask = "Muon_ZVeto",
                     inmask = "Muon_looseMask", vals = Muon.muon_part,
                     addvals = [("Muon_finalMask", "Muon_looseIndex")])

        self.add_job("lep_sf", outmask = "Muon_scale", inmask = "Muon_looseMask",
                     vals = pre("Muon", ["pt", "eta"]))
        self.add_job("lep_tracking_sf", outmask = "Muon_trackingScale",
                     inmask = "Muon_looseMask", vals = ["Muon_eta"])
    # Numba methods

    loose = pre("Muon", ["pt", "eta", "isGlobal", "isTracker", "isPFcand",
                         'miniPFRelIso_all', 'dz', 'dxy'])
    @staticmethod
    @numba.vectorize('b1(f4,f4,b1,b1,b1,f4,f4,f4)')
    def loose_mask(pt, eta, isGlobal, isTracker, isPFcand, iso, dz, dxy):
        return (pt > 5 and
           abs(eta) < 2.4 and
           (isGlobal or isTracker) and
           isPFcand and
           iso < 0.4 and
           abs(dz) < 0.1 and
           abs(dxy) < 0.05
           )

    fake = pre("Muon", ["pt", "tightCharge", "mediumId", "sip3d"])
    @staticmethod
    @numba.vectorize('b1(f4,i4,b1,f4)')
    def fake_mask(pt, tightCharge, mediumId, sip3d):
        return (
            pt > 10 and
            tightCharge == 2 and
            mediumId and
            sip3d < 4
           )

    tight = pre("Muon", ["pt", 'eta', 'miniPFRelIso_all'])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4)')
    def tight_mask(pt, eta, iso):
        return (
            #pt > 20 and
            pt > 15 and
            iso < 0.16
           )
    
    @staticmethod
    @numba.jit(nopython=True)
    def looseIdx(events, builder):
        for event in events:
            builder.begin_list()
            for midx in range(len(event.Muon_pt)):
                builder.integer(midx)
            builder.end_list()

    close_jet = ["Muon_eta", "Muon_phi", "Jet_eta", "Jet_phi"]
    @staticmethod
    @numba.jit(nopython=True)
    def closeJet(events, builder):
        for event in events:
            builder.begin_list()
            for midx in range(len(event.Muon_eta)):
                mindr = 10 # 0.16  # 0.4**2
                minidx = -1
                for jidx in range(len(event.Jet_eta)):
                    dr = deltaR(event.Muon_phi[midx], event.Muon_eta[midx],
                                event.Jet_phi[jidx], event.Jet_eta[jidx])
                    if mindr > dr:
                        mindr = dr
                        minidx = jidx
                builder.begin_list()
                builder.integer(minidx)
                builder.real(mindr)
                builder.end_list()
            builder.end_list()

    v_fullIso = pre("Muon", ["pt", "eta", "phi"]) + pre("Jet", ["pt", "eta", "phi"])
    @staticmethod
    @numba.jit(nopython=True)
    def fullIso(events, builder):
        I2 = 0.76
        I3_pow2 = 7.2**2
        for event in events:
            builder.begin_list()
            for midx in range(len(event.Muon_eta)):
                jidx = int(event.Muon_closeJetIndex[midx][0])
                if jidx < 0 or event.Muon_pt[midx]/event.Jet_pt[jidx] > I2:
                    builder.boolean(True)
                    continue
                jetrel = jetRel(event.Muon_pt[midx], event.Muon_eta[midx],
                                event.Muon_phi[midx], event.Jet_pt[jidx],
                                event.Jet_eta[jidx], event.Jet_phi[jidx])
                builder.boolean(jetrel > I3_pow2)
            builder.end_list()

    muon_part = pre("Muon", ["pt", "eta", "phi", "charge"])
    @staticmethod
    @numba.jit(nopython=True)
    def pass_zveto(events, builder):
        for event in events:
            passed = True
            for idx in range(len(event.Muon_looseIndex)):
                i = event.Muon_looseIndex[idx]

                for j in range(len(event.Muon_pt)):
                    if event.Muon_charge[i]*event.Muon_charge[j] > 0:
                        continue
                    if in_zmass(event.Muon_pt[i], event.Muon_eta[i],
                                event.Muon_phi[i], event.Muon_pt[j],
                                event.Muon_eta[j], event.Muon_phi[j]):
                        passed = False
                        break
                if not passed:
                    break
            builder.boolean(passed)

    @staticmethod
    @numba.vectorize('f4(f4,f4)')
    def lep_sf(pt, eta):
        sf = np.array([[0.9047, 0.8860, 0.8916, 0.8394],
                       [0.9430, 0.9685, 0.9741, 0.8917],
                       [0.9707, 0.9724, 0.9777, 0.9180],
                       [0.9821, 0.9850, 0.9934, 0.9389],
                       [0.9854, 0.9861, 0.9968, 0.9453],
                       [0.9813, 0.9819, 0.9964, 0.9410],
                       [0.9830, 0.9861, 0.9994, 0.9525]])
        pt_edges = np.array([20, 25, 30, 40, 50, 60, 14000])
        eta_edges = np.array([0.9, 1.2, 2.1, 2.4])

        return sf[np.argmax(pt <= pt_edges), np.argmax(abs(eta) <= eta_edges)]

    @staticmethod
    @numba.vectorize('f4(f4)')
    def lep_tracking_sf(eta):
        sf = np.array([0.9879, 0.9939, 0.9970, 0.9954, 0.9937, 0.9959, 0.9976,
                       0.9961, 0.9930, 0.9819])
        eta_edges = np.array([-2.1, -1.6, -1.1, -0.6, 0.0, 0.6, 1.1, 1.6,
                              2.1, 2.4])
        return sf[np.argmax(eta <= eta_edges)]
