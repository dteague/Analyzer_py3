#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba
import math

from python.Process import Process
from Utilities.FileGetter import pre
from python.Common import deltaR, jetRel, in_zmass

class Electron(Process):
    def __init__(self, process):
        super().__init__(process)

        self.add_job("loose_mask", outmask = "Electron_basicLooseMask",
                     vals = Electron.loose)
        self.add_job("trigger_emu", outmask = "Electron_triggerEmuMask",
                     inmask = "Electron_basicLooseMask", vals = Electron.emu)
        self.add_job("mva_loose_2016", outmask = "Electron_looseMask",
                     inmask = "Electron_triggerEmuMask",
                     vals = Electron.mva + Electron.mva_2016)

        self.add_job("looseIdx", outmask = "Electron_looseIndex",
                     inmask = "Electron_looseMask",
                     vals = ["Electron_pt"])

        self.add_job("fake_mask", outmask = "Electron_basicFakeMask",
                     inmask = "Electron_looseMask", vals = Electron.fake)
        self.add_job("closeJet", outmask = "Electron_closeJetIndex",
                     inmask = "Electron_basicFakeMask", vals = Electron.close_jet)
        self.add_job("fullIso", outmask = "Electron_fakeMask",
                     inmask = "Electron_basicFakeMask", vals = Electron.v_fullIso,
                     addvals = {"Electron_closeJetIndex": None})

        self.add_job("tight_mask", outmask = "Electron_basicTightMask",
                     inmask = "Electron_fakeMask", vals = Electron.tight)
        self.add_job("mva_tight_2016", outmask = "Electron_finalMask",
                     inmask = "Electron_basicTightMask",
                     vals = Electron.mva + Electron.mva_2016)

        self.add_job("pass_zveto", outmask = "Electron_ZVeto",
                     inmask = "Electron_looseMask", vals = Electron.elec_part,
                     addvals = {"Electron_looseIndex": "Electron_finalMask"})

    
    mva =  pre("Electron", ["pt", "eCorr", "eta"])
    mva_2016 = ["Electron_mvaSpring16GP"]
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4)',nopython=True)
    def mva_loose_2016(pt, eCorr, eta, mva):
        A = np.array([-0.48, -0.67, -0.49])
        B = np.array([-0.85, -0.91, -0.83])
        C = (B-A)/10
        if pt/eCorr < 5: return False
        elif pt/eCorr < 10:  mvaVec = np.array([-0.46, -0.03, 0.06])
        elif pt/eCorr < 15:  mvaVec = A
        elif pt/eCorr < 25:  mvaVec = A + C*(pt/eCorr-15)
        else:                mvaVec = B
        
        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]
        
        return mva > mvaCut

    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4)',nopython=True)
    def mva_tight_2016(pt, eCorr, eta, mva):
        A = np.array([0.77, 0.56, 0.48])
        B = np.array([0.52, 0.11, -0.01])
        C = (B-A)/10
        if pt/eCorr < 10: return False
        elif pt/eCorr < 15:  mvaVec = A
        elif pt/eCorr < 25:  mvaVec = A + C*(pt/eCorr-15)
        else:                mvaVec = B

        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return mva > mvaCut
    
    mva_2017 = ["Electron_mvaFall17V1noIso"]
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4)')
    def mva_loose_2017(pt, eCorr, eta, mva):
        A = np.array([0.488, -0.045, 0.176])
        B = np.array([-0.64, -0.775, -0.733])
        C = np.array([0.148, 0.075, 0.077])
        if pt/eCorr < 5: return False
        elif pt/eCorr < 10:  mvaVec = A
        elif pt/eCorr < 25:  mvaVec = B - C*(1 - (pt/eCorr-10)/15)
        else:                mvaVec = B
        
        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return mva > mvaCut

    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4)')
    def mva_tight_2017(pt, eCorr, eta, mva):
        B = np.array([0.68, 0.475, 0.32])
        C = np.array([0.48 , 0.375, 0.42])
        if pt/eCorr < 10: return False
        elif pt/eCorr < 25:  mvaVec = B - C*(1 - (pt/eCorr-10)/15)
        else:                mvaVec = B
        
        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return mva > mvaCut
    
    mva_2018 = ["Electron_mvaFall17V2noIso"]
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4)')
    def mva_loose_2018(pt, eCorr, eta, mva):
        A = np.array([1.32, 0.192, 0.363])
        B = np.array([1.204, 0.084, -0.123])
        C = np.array([0.066, 0.033, 0.053])
        if pt < 5: return False

        elif pt/eCorr < 10:  mvaVec = A
        elif pt/eCorr < 25:  mvaVec = B + C*(pt/eCorr-25)
        else:                mvaVec = B

        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return math.atanh(mva) > mvaCut

    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4)')
    def mva_tight_2018(pt, eCorr, eta, mva):
        B = np.array([4.277, 3.152, 2.359])
        C = np.array([0.112, 0.06, 0.087])
        if pt/eCorr < 10: return False
        elif pt/eCorr < 25:  mvaVec = B + C*(pt/eCorr-25)
        else:                mvaVec = B
        
        if abs(eta) < 0.8:     mvaCut = mvaVec[0]
        elif abs(eta) < 1.479: mvaCut = mvaVec[1]
        elif abs(eta) < 2.5:   mvaCut = mvaVec[2]

        return math.atanh(mva) > mvaCut

    # Numba methods
    loose = pre("Electron", ["pt", "eCorr", "eta", "convVeto", "lostHits",
                             "miniPFRelIso_all", "dz", "dxy"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,b1,u1,f4,f4,f4)')
    def loose_mask(pt, eCorr, eta, convVeto, lostHits, iso, dz, dxy):
        return (
            pt/eCorr > 7 and
            abs(eta) < 2.5 and
            convVeto  and
            lostHits <= 1 and
            iso < 0.4 and
            abs(dz) < 0.1 and
            abs(dxy) < 0.05
           )

    emu = pre("Electron", ["eta", "sieie", "hoe", "eInvMinusPInv"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4)')
    def trigger_emu(eta, sieie, hoe, eInvMinusPInv):
        passed = (abs(eInvMinusPInv) < 0.01 and hoe < 0.08)
        if passed:
            return ((abs(eta) < 1.479 and sieie < 0.011) or
               (abs(eta) >= 1.479 and sieie < 0.031))
        else:
            return False

    fake = pre("Electron", ["pt", "eCorr", "sip3d", "tightCharge", "lostHits"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,i4,u1)')
    def fake_mask(pt, eCorr, sip3d, tightCharge, lostHits):
        return (
            pt/eCorr >= 10 and
            sip3d < 4 and
            lostHits == 0 and
            tightCharge == 2
        )


    tight = pre("Electron", ["pt", "eCorr", "miniPFRelIso_all",
                             "dr03EcalRecHitSumEt", "dr03HcalDepth1TowerSumEt",
                             "dr03TkSumPt"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,f4,f4,f4,f4)')
    def tight_mask(pt, eCorr, iso, EcalSumEt, HcalSumEt, TkSumPt):
        pt_cor = pt/eCorr
        return (
            # pt/eCorr > 20 and
            pt_cor > 15 and
            iso < 0.12 and
            EcalSumEt / pt_cor < 0.45 and
            HcalSumEt / pt_cor < 0.25 and
            TkSumPt / pt_cor < 0.2
        )

    @staticmethod
    @numba.jit(nopython=True)
    def looseIdx(events, builder):
        for event in events:
            builder.begin_list()
            for eidx in range(len(event.Electron_pt)):
                builder.integer(eidx)
            builder.end_list()

    close_jet = ["Electron_eta", "Electron_phi", "Jet_eta", "Jet_phi"]
    @staticmethod
    @numba.jit(nopython=True)
    def closeJet(events, builder):
        i = 0
        for event in events:
            builder.begin_list()
            for eidx in range(len(event.Electron_eta)):
                mindr = 10
                minidx = -1
                for jidx in range(len(event.Jet_eta)):
                    dr = deltaR(event.Electron_phi[eidx], event.Electron_eta[eidx],
                                event.Jet_phi[jidx], event.Jet_eta[jidx])
                    if mindr > dr:
                        mindr = dr
                        minidx = jidx
                builder.begin_list()
                builder.integer(minidx)
                builder.real(mindr)
                builder.end_list()
            builder.end_list()
            i += 1
    
    v_fullIso = pre("Electron", ["pt", "eCorr", "eta", "phi"]) + \
        ["Jet_pt", "Jet_eta", "Jet_phi"]
    @staticmethod
    @numba.jit(nopython=True)
    def fullIso(events, builder):
        I2 = 0.8
        I3_pow2 = 7.2**2
        i = 0
        for event in events:
            builder.begin_list()
            for eidx in range(len(event.Electron_eta)):
                jidx = int(event.Electron_closeJetIndex[eidx][0])
                pt = event.Electron_pt[eidx] / event.Electron_eCorr[eidx]
                if pt/event.Jet_pt[jidx] > I2:
                    builder.boolean(True)
                    continue
                jetrel = jetRel(pt, event.Electron_eta[eidx],
                                event.Electron_phi[eidx], event.Jet_pt[jidx],
                                event.Jet_eta[jidx], event.Jet_phi[jidx])
                builder.boolean(jetrel > I3_pow2)
            builder.end_list()
            i += 1

    elec_part = pre("Electron", ["pt", "eCorr", "eta", "phi", "charge"])
    @staticmethod
    @numba.jit(nopython=True)
    def pass_zveto(events, builder):
        for event in events:
            passed = True
            for idx in range(len(event.Electron_looseIndex)):
                i = event.Electron_looseIndex[idx]
                pt_i = event.Electron_pt[i]/event.Electron_eCorr[i]
                for j in range(len(event.Electron_pt)):
                    if event.Electron_charge[i]*event.Electron_charge[j] > 0:
                        continue
                    pt_j = event.Electron_pt[j]/event.Electron_eCorr[j]
                    if in_zmass(pt_i, event.Electron_eta[i], event.Electron_phi[i],
                                pt_j, event.Electron_eta[j], event.Electron_phi[j]):
                        passed = False
                        break
                if not passed:
                    break
            builder.boolean(passed)
