#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from python.Process import Process

class Muon(Process):
    def __init__(self):
        super().__init__()

        self.extraFuncs = [
            ("loose_mask", "Muon_looseMask", None, Muon.loose),
            ("fake_mask", "Muon_fakeMask", None, Muon.fake),
            ("closeJet", "Muon_closeJetIndex", {"Muon_": "Muon_fakeMask"}, Muon.close_jet),
            ("tight_mask", "Muon_tightMask", "Muon_fakeMask", Muon.tight),
        ]

    # Numba methods

    loose = Process.prefix("Muon", ["pt", "isGlobal", "isTracker", "isPFcand",
                                'miniPFRelIso_all', 'dxy', 'dz'])
    @staticmethod
    @numba.vectorize('b1(f4,b1,b1,b1,f4,f4,f4)')
    def loose_mask(pt, isGlobal, isTracker, isPFcand, iso, dz, dxy):
        return (pt > 5 and
           (isGlobal or isTracker) and
           isPFcand and
           iso < 0.4 and
           np.abs(dz) < 0.1 and
           np.abs(dz) < 0.05
           )

    fake = Process.prefix("Muon", [ "tightCharge", "mediumId", "sip3d"])
    @staticmethod
    @numba.vectorize('b1(i4,b1,f4)')
    def fake_mask(tightCharge, mediumId, sip3d):
        return (
            tightCharge == 2 and
            mediumId and
            sip3d < 4
           )

    tight = Process.prefix("Muon", ["pt", 'miniPFRelIso_all'])
    @staticmethod
    @numba.vectorize('b1(f4,f4)')
    def tight_mask(pt, iso):
        return (
            pt > 20 and
            iso < 0.16
           )

    close_jet = ["Muon_eta", "Muon_phi", "Jet_eta", "Jet_phi"]
    def setup_closeJet(self, events):
        #muons = events[["Muon_eta", "Muon_phi"]][self.masks["Muon_fakeableMask"]]
        meta, jeta = ak.unzip(ak.cartesian([events.Muon_eta, events.Jet_eta], nested=True, axis=1))
        mphi, jphi = ak.unzip(ak.cartesian([events.Muon_phi, events.Jet_phi], nested=True, axis=1))

        diff = (meta - jeta)**2 + (mphi - jphi)**2
        return ak.argmin(diff, axis=2)

    @staticmethod
    @numba.jit(nopython=True)
    def closeJet(events, builder):
        for event in events:
            builder.begin_list()
            for midx in range(len(event.Muon_eta)):
                mindr = 0.16  # 0.4**2
                minidx = -1
                for jidx in range(len(event.Jet_eta)):
                    dr = (event.Muon_eta[midx] - event.Jet_eta[jidx])**2 + \
                        (event.Muon_phi[midx] - event.Jet_phi[jidx])**2
                    if mindr > dr:
                        mindr = dr
                        minidx = jidx
                builder.integer(minidx)
            builder.end_list()
