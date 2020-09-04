#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba
import math

from python.Process import Process
from Utilities.FileGetter import pre

class Muon(Process):
    def __init__(self, process):
        super().__init__(process)

        self.add_job("loose_mask", outmask = "Muon_looseMask",
                     vals = Muon.loose)
        self.add_job("looseIdx", outmask = "Muon_looseIndex",
                     vals = ["Muon_pt"])
        self.add_job("fake_mask", outmask = "Muon_fakeMask",
                     inmask = "Muon_looseMask", vals = Muon.fake)
        self.add_job("closeJet", outmask = "Muon_closeJetIndex",
                     inmask = "Muon_fakeMask", vals = Muon.close_jet)
        self.add_job("tight_mask", outmask = "Muon_tightMask",
                     inmask = "Muon_fakeMask", vals = Muon.tight)
        self.add_job("fullIso", outmask = "Muon_finalMask",
                     inmask = "Muon_tightMask", vals = Muon.v_fullIso,
                     addvals = {"Muon_closeJetIndex": "Muon_tightMask"})
        self.add_job("pass_zveto", outmask = "Muon_ZVeto",
                     inmask = "Muon_looseMask", vals = Muon.muon_part,
                     addvals = {"Muon_looseIndex": "Muon_finalMask"})

    # Numba methods

    loose = pre("Muon", ["pt", "isGlobal", "isTracker", "isPFcand",
                         'miniPFRelIso_all', 'dxy', 'dz'])
    @staticmethod
    @numba.vectorize('b1(f4,b1,b1,b1,f4,f4,f4)')
    def loose_mask(pt, isGlobal, isTracker, isPFcand, iso, dz, dxy):
        return (pt > 5 and
           (isGlobal or isTracker) and
           isPFcand and
           iso < 0.4 and
           abs(dz) < 0.1 and
           abs(dz) < 0.05
           )

    fake = pre("Muon", ["tightCharge", "mediumId", "sip3d"])
    @staticmethod
    @numba.vectorize('b1(i4,b1,f4)')
    def fake_mask(tightCharge, mediumId, sip3d):
        return (
            tightCharge == 2 and
            mediumId and
            sip3d < 4
           )

    tight = pre("Muon", ["pt", 'miniPFRelIso_all'])
    @staticmethod
    @numba.vectorize('b1(f4,f4)')
    def tight_mask(pt, iso):
        return (
            pt > 20 and
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
                    dr = (event.Muon_eta[midx] - event.Jet_eta[jidx])**2 + \
                        (event.Muon_phi[midx] - event.Jet_phi[jidx])**2
                    if mindr > dr:
                        mindr = dr
                        minidx = jidx
                builder.integer(minidx)
            builder.end_list()

    v_fullIso = pre("Muon", ["pt", "eta", "phi"]) + pre("Jet", ["pt", "eta", "phi"])
    @staticmethod
    @numba.jit(nopython=True)
    def fullIso(events, builder):
        I2 = 0.72
        I3_pow2 = 7.2**2
        for event in events:
            builder.begin_list()
            for midx in range(len(event.Muon_eta)):
                jidx = event.Muon_closeJetIndex[midx]
                if event.Muon_pt[midx]/event.Jet_pt[jidx] > I2:
                    builder.boolean(True)
                    continue
                p_jet = event.Jet_pt[jidx]*math.cosh(event.Jet_eta[jidx])
                p_mu = event.Muon_pt[midx]*math.cosh(event.Muon_eta[midx])
                cos_dphi = (event.Jet_pt[jidx]*event.Muon_pt[midx]
                            *math.cos(event.Jet_eta[jidx]-event.Muon_eta[midx]))
                jetrel = (((p_jet*p_mu)**2 - p_jet*p_mu - cos_dphi)/
                          (p_jet-p_mu)**2 - cos_dphi)
                builder.boolean(jetrel > I3_pow2)
            builder.end_list()

    muon_part = pre("Muon", ["pt", "eta", "phi", "charge"])
    @staticmethod
    @numba.jit(nopython=True)
    def pass_zveto(events, builder):
        for event in events:
            passed = True
            for i in range(len(event.Muon_pt)):
                for j in range(i+1, len(event.Muon_pt)):
                    if event.Muon_charge[i]*event.Muon_charge[j] > 0:
                        continue
                    ptval = 2*event.Muon_pt[i]*event.Muon_pt[j]
                    mass = ptval*(math.cosh(event.Muon_eta[i] - event.Muon_eta[j])
                                  - math.cos(event.Muon_phi[i] - event.Muon_phi[j]))
                    if mass < 12 or abs(mass - 91.188) < 15:
                        for k in event.Muon_looseIndex:
                            if i == k or j == k:
                                passed = False
                    if not passed:
                        break
                if not passed:
                    break
            builder.boolean(passed)
