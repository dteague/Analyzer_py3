#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba
import math

from python.Process import Process
from Utilities.FileGetter import pre

class EventWide(Process):
    def __init__(self, process):
        super().__init__(process)

        self.add_job("met_filter", outmask = "Event_MetFilterMask",
                     vals = EventWide.filters)
        self.add_job("pileup_scale", outmask = "Event_pileupScale",
                     vals = ["Pileup_nTrueInt"])
        self.add_job("wdecay_scale", outmask = "Event_wDecayScale",
                     vals = EventWide.gen_vars)
        self.add_job("set_channel", outmask = "Event_channels",
                     inmask = ["Electron_finalMask", "Muon_finalMask"],
                     vals = EventWide.lep_chans)
        self.add_job("apply_trigger", outmask = "Event_triggerMask",
                     vals = EventWide.triggers,
                     addvals = {"Event_channels": None})
        self.add_job("calc_HT", outmask = "Event_HT", inmask = "Jet_jetMask",
                     vals = EventWide.ht)
        self.add_job("calc_centrality", outmask = "Event_centrality",
                     inmask = "Jet_jetMask", vals = EventWide.centrality,
                     addvals = {"Event_HT": None})
        self.add_job("calc_sphericity", outmask = "Event_sphericity",
                     inmask = "Jet_jetMask", vals = EventWide.sphericity)
        
    # Numba methods
    # maybe just Flag_MetFilter?
    filters = pre("Flag",["goodVertices", "globalSuperTightHalo2016Filter",
                          "HBHENoiseFilter", "HBHENoiseIsoFilter",
                          "EcalDeadCellTriggerPrimitiveFilter",
                          "BadPFMuonFilter", "ecalBadCalibFilter"])
    @staticmethod
    @numba.vectorize('b1(b1,b1,b1,b1,b1,b1,b1)')
    def met_filter(goodVertices, TightHaloFilter, HBHEFilter, HBHEIsoFilter,
                 ecalDeadCellFilter, badMuonFilter, ecalBadCalibFilter):
        return (
            goodVertices and
            TightHaloFilter and
            HBHEFilter and
            HBHEIsoFilter and
            ecalDeadCellFilter and
            badMuonFilter and
            ecalBadCalibFilter

        )


    @staticmethod
    @numba.vectorize('f8(f8)', nopython=True)
    def pileup_scale(pileup):
        pileupScales = [
            0.366077, 0.893925, 1.197716, 0.962699, 1.120976, 1.164859, 0.795599, 0.495824,
            0.742182, 0.878856, 0.964232, 1.072499, 1.125335, 1.176027, 1.202083, 1.207643,
            1.200176, 1.182682, 1.143998, 1.096632, 1.065602, 1.051166, 1.051600, 1.050630,
            1.049862, 1.058173, 1.072155, 1.083030, 1.095693, 1.107871, 1.094621, 1.082620,
            1.041247, 0.985752, 0.910807, 0.820923, 0.716787, 0.610013, 0.503118, 0.404841,
            0.309195, 0.227920, 0.163690, 0.113180, 0.077300, 0.050922, 0.031894, 0.020094,
            0.012263, 0.007426, 0.004380, 0.002608, 0.001566, 0.000971, 0.000729, 0.000673,
            0.000730, 0.000949, 0.001355, 0.001894, 0.003082, 0.004097, 0.004874, 0.005256,
            0.005785, 0.005515, 0.005000, 0.004410, 0.004012, 0.003548, 0.003108, 0.002702,
            0.002337, 0.002025, 0.001723, ]
        if int(pileup) < len(pileupScales):
            return 0.
        else:
            return pileupScales[int(pileup)]


    gen_vars = pre("GenPart", ["pdgId", "genPartIdxMother", "status"]) \
        + ["nGenPart"]
    @staticmethod
    @numba.jit(nopython=True)
    def wdecay_scale(events, builder):
        pdglepW = 0.3258
        genlepW = 1.0/3
        lep_ratio = pdglepW/genlepW
        had_ratio = (1 - pdglepW)/(1 - genlepW)

        for event in events:
            nlepW = 0
            nW = 0
            for i in range(event.nGenPart):
                if (abs(event.GenPart_pdgId[i]) == 24 and
                    (event.GenPart_status[i] == 22 or event.GenPart_status[i] == 52) and
                    abs(event.GenPart_pdgId[event.GenPart_genPartIdxMother[i]]) != 24):

                    nW += 1
                elif((abs(event.GenPart_pdgId[i]) == 12
                       or abs(event.GenPart_pdgId[i]) == 14
                       or abs(event.GenPart_pdgId[i]) == 16)
                     and abs(event.GenPart_pdgId[event.GenPart_genPartIdxMother[i]]) == 24):
                     nlepW += 1

            nhadW = nW - nlepW
            builder.real(lep_ratio**nlepW * had_ratio**nhadW)

    lep_chans = ["Electron_charge", "Electron_pt", "Muon_charge", "Muon_pt"]
    @staticmethod
    @numba.jit(nopython=True)
    def set_channel(events, builder):
        for event in events:
            nLeps = len(event.Electron_charge) + len(event.Muon_charge)
            if nLeps <= 1:
                builder.integer(nLeps)
            else:
                q1, q2 = 0, 0
                p1, p2 = 0, 0
                chan = 10*nLeps
                if len(event.Electron_pt) > 1:
                    p2, q2 = event.Electron_pt[1], event.Electron_charge[1]
                if len(event.Electron_pt) > 0:
                    p1, q1 = event.Electron_pt[0], event.Electron_charge[0]
                if len(event.Muon_pt) > 0:
                    if event.Muon_pt[0] > p1:
                        p2, q2 = p1, q2
                        p1, q1 = event.Muon_pt[0], event.Muon_charge[0]
                        chan += 1
                    elif event.Muon_pt[0] > p2:
                        p2, q2 = event.Muon_pt[0], event.Muon_charge[0]
                        chan += 1
                if len(event.Muon_pt) > 1:
                    if event.Muon_pt[1] > p2:
                        p2, q2 = event.Muon_pt[0], event.Muon_charge[0]
                        chan += 1
                builder.integer(chan*q1*q2)

    triggers = pre("HLT", ["DoubleMu8_Mass8_PFHT300",
                           "Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT300",
                           "DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT300",
                           "AK8PFJet450", "PFJet450"])
    @staticmethod
    @numba.vectorize('b1(b1,b1,b1,b1,b1,i8)')
    def apply_trigger(mm_trig, em_trig, ee_trig, ak8ht450_trig, ht450_trig, chan):
        return (
            (abs(chan) <= 1) or
            ((abs(chan) % 3 == 0 and ee_trig) or
            (abs(chan) % 3 == 1 and em_trig) or
            (abs(chan) % 3 == 2 and mm_trig)) or
            (ht450_trig or ak8ht450_trig)
        )
    
    ht = ["Jet_pt"]
    @staticmethod
    @numba.jit(nopython=True)
    def calc_HT(events, builder):
        for event in events:
            HT = 0
            for j in range(len(event["Jet_pt"])):
                HT += event.Jet_pt[j]
            builder.real(HT)
            
    centrality = ["Jet_pt", "Jet_eta", "Jet_mass"]
    @staticmethod
    @numba.jit(nopython=True)
    def calc_centrality(events, builder):
        for event in events:
            eTot = 0.0001 # for 0jet case
            for j in range(len(event["Jet_pt"])):
                p = event.Jet_pt[j]*math.cosh(event.Jet_eta[j])
                eTot += math.sqrt(p**2 + event.Jet_mass[j])
            builder.real(event.Event_HT/eTot)

    sphericity = ["Jet_pt", "Jet_eta", "Jet_phi"]
    @staticmethod
    @numba.jit(nopython=True)
    def calc_sphericity(events, builder):
        for event in events:
            if len(event["Jet_pt"]) == 0:
                builder.real(-1)
                continue
            sphere = np.zeros((3,3))
            for i in range(len(event["Jet_pt"])):
                p_vec = [event.Jet_pt[i]*math.cos(event.Jet_phi[i]),
                         event.Jet_pt[i]*math.sin(event.Jet_phi[i]),
                         event.Jet_pt[i]*math.sinh(event.Jet_eta[i])]
                sphere += np.outer(p_vec, p_vec)
            sphere = sphere/np.trace(sphere)
            eig, _ = np.linalg.eig(sphere)
            builder.real(3/2*(1-max(eig)))



    # close_jet = ["Muon_eta", "Muon_pt", "Electron_eta", "Electron_pt"]
    # @staticmethod
    # @numba.jit(nopython=True)
    # def trigger_scale(events, builder):
    #     for event in events:
    #         if len(event.Muon_pt) + len(event.Electron_pt) < 2:
    #             builder.real(1)
    #             continue

    #         if

            # float triggerScaleFactor(int pdgId1, int pdgId2, float pt1, float pt2, float eta1, float eta2, float ht) {
#     // return TotalTriggerSF(pdgId1, pt1, eta1, pdgId2, pt2, eta2, ht);
#     // Using Matthieu's macro, so dummy 1 here
#     return 1.0; // FIXME


#     if (ht>300) {
# 	if ((abs(pdgId1)+abs(pdgId2))==22) return 1.;
# 	if ((abs(pdgId1)+abs(pdgId2))==26) return 0.985*0.985;
# 	if ((abs(pdgId1)+abs(pdgId2))==24) return 0.985;
#     } else {
# 	if ((abs(pdgId1)+abs(pdgId2))==22) return 0.997*0.997*0.998;
# 	if ((abs(pdgId1)+abs(pdgId2))==26) return 0.982*0.985*0.973;
# 	if ((abs(pdgId1)+abs(pdgId2))==24) {
# 	    if (abs(pdgId1)==11) {
# 		if (pt1>pt2) return 0.997*0.985;
# 		else return 0.997*0.982;
# 	    } else {
# 		if (pt1>pt2) return 0.997*0.982;
# 		else return 0.997*0.985;
# 	    }
# 	}
#     }
#     return 0.;
# }
