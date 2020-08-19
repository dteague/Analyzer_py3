#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from python.Process import Process

class Electron(Process):
    def __init__(self):
        super().__init__()

        self.extraFuncs = [
            ("loose_mask", "Electron_looseMask", None, Electron.loose),
            ("trigger_emu", "Electron_looseMask", "Electron_looseMask", Electron.emu),
            ("mva_loose_2016", "Electron_looseMask", "Electron_looseMask", Electron.mva_2016),
            ("fake_mask", "Electron_fakeMask", "Electron_looseMask", Electron.fake),
            ("closeJet", "Electron_closeJetIndex", {"Electron_": "Electron_fakeMask"}, Electron.close_jet),
            ("tight_mask", "Electron_tightMask", "Electron_fakeMask", Electron.tight),
            ("mva_tight_2016", "Electron_tightMask", "Electron_tightMask", Electron.mva_2016),
        ]

    mva_2016 = Process.prefix("Electron", ["pt", "eta", "mvaFall17V2noIso"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4)',nopython=True)
    def mva_loose_2016(pt, eta, mva):
        A = np.array([-0.48, -0.67, -0.49])
        B = np.array([-0.85, -0.91, -0.83])
        C = (B-A)/10
        if pt < 5: return False
        elif pt < 10:  mvaVec = np.array([-0.46, -0.03, 0.06])
        elif pt < 15:  mvaVec = A
        elif pt < 25:  mvaVec = A + C*(pt-15)
        else:          mvaVec = B
        
        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return mva > mvaCut

    @staticmethod
    @numba.vectorize('b1(f4,f4,f4)',nopython=True)
    def mva_tight_2016(pt, eta, mva):
        A = np.array([0.77, 0.56, 0.48])
        B = np.array([0.52, 0.11, -0.01])
        C = (B-A)/10
        if pt < 10: return False
        elif pt < 15:  mvaVec = A
        elif pt < 25:  mvaVec = A + C*(pt-15)
        else:          mvaVec = B

        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return mva > mvaCut

    mva_2017 = Process.prefix("Electron", ["pt", "eta", "mvaFall17V1noIso"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4)')
    def mva_loose_2017(pt, eta, mva):
        A = np.array([0.488, -0.045, 0.176])
        B = np.array([-0.64, -0.775, -0.733])
        C = np.array([0.148, 0.075, 0.077])
        if pt < 5: return False
        elif pt < 10:  mvaVec = A
        elif pt < 25:  mvaVec = B - C*(1 - (pt-10)/15)
        else:          mvaVec = B
        
        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return mva > mvaCut

    @staticmethod
    @numba.vectorize('b1(f4,f4,f4)')
    def mva_tight_2017(pt, eta, mva):
        B = np.array([0.68, 0.475, 0.32])
        C = np.array([0.48 , 0.375, 0.42])
        if pt < 10: return False
        elif pt < 25:  mvaVec = B - C*(1 - (pt-10)/15)
        else:          mvaVec = B
        
        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return mva > mvaCut

    mva_2016 = Process.prefix("Electron", ["pt", "eta", "mvaSpring16GP"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4)')
    def mva_loose_2018(pt, eta, mva):
        A = np.array([1.32, 0.192, 0.363])
        B = np.array([1.204, 0.084, -0.123])
        C = np.array([0.066, 0.033, 0.053])
        if pt < 5: return False
        elif pt < 10:  mvaVec = A
        elif pt < 25:  mvaVec = B + C*(pt-25)
        else:          mvaVec = B

        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return np.arctanh(mva) > mvaCut

    @staticmethod
    @numba.vectorize('b1(f4,f4,f4)')
    def mva_tight_2018(pt, eta, mva):
        B = np.array([4.277, 3.152, 2.359])
        C = np.array([0.112, 0.06, 0.087])
        if pt < 10: return False
        elif pt < 25:  mvaVec = B + C*(pt-25)
        else:          mvaVec = B
        
        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return np.arctanh(mva) > mvaCut

    # Numba methods
    loose = Process.prefix("Electron", ["pt", "convVeto", "lostHits",
                                        "miniPFRelIso_all", "dz", "dxy"])
    @staticmethod
    @numba.vectorize('b1(f4,b1,u1,f4,f4,f4)')
    def loose_mask(pt, convVeto, lostHits, iso, dz, dxy):
        return (
            pt > 7 and
            convVeto  and
            lostHits <= 1 and
            iso < 0.4 and
            np.abs(dz) < 0.1 and
            np.abs(dxy) < 0.05
           )

    emu = Process.prefix("Electron", ["eta", "sieie", "hoe", "eInvMinusPInv"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4)')
    def trigger_emu(eta, sieie, hoe, eInvMinusPInv):
        passed = (eInvMinusPInv < 0.01 and hoe < 0.08)
        if passed:
            return ((abs(eta) < 1.479 and sieie < 0.011) or
               (abs(eta) >= 1.479 and sieie < 0.031))
        else:
            return False

    fake = Process.prefix("Electron", ["sip3d", "tightCharge", "lostHits"])
    @staticmethod
    @numba.vectorize('b1(f4,i4,u1)')
    def fake_mask(sip3d, tightCharge, lostHits):
        return (
            sip3d < 4 and
            lostHits == 0 and
            tightCharge == 2
        )


    tight = Process.prefix("Electron",
                        ["pt", "miniPFRelIso_all", "dr03EcalRecHitSumEt",
                         "dr03HcalDepth1TowerSumEt", "dr03TkSumPt"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4,f4)')
    def tight_mask(pt, iso, EcalSumEt, HcalSumEt, TkSumPt):
        return (
            pt > 20 and
            iso < 0.12 and
            EcalSumEt / pt < 0.45 and
            HcalSumEt / pt < 0.25 and
            TkSumPt / pt < 0.2
        )

    close_jet = ["Electron_eta", "Electron_phi", "Jet_eta", "Jet_phi"]
    @staticmethod
    @numba.jit(nopython=True)
    def closeJet(events, builder):
        for event in events:
            builder.begin_list()
            for midx in range(len(event.Electron_eta)):
                mindr = 0.16  # 4**2
                minidx = -1
                for jidx in range(len(event.Jet_eta)):
                    dr = (event.Electron_eta[midx] - event.Jet_eta[jidx])**2 \
                        + (event.Electron_phi[midx] - event.Jet_phi[jidx])**2
                    if mindr > dr:
                        mindr = dr
                        minidx = jidx
                builder.integer(minidx)
            builder.end_list()
