#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from python.Process import Process





class EventWide(Process):
    def __init__(self, process):
        super().__init__(process)
        self.extraFuncs = [
            ("met_filter", "Event_MetFilterMask", None, EventWide.filters),
            ("pileup_scale", "Event_pileupScale", None, ["Pileup_nTrueInt"]),
            ("wdecay_scale", "Event_wDecayScale", None, EventWide.gen_vars),
        ]

    # Numba methods
    # maybe just Flag_MetFilter?
    filters = Process.prefix("Flag",
                            ["goodVertices", "globalSuperTightHalo2016Filter",
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


    gen_vars = Process.prefix("GenPart", ["pdgId", "genPartIdxMother",
                                          "status"]) + ["nGenPart"]
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
