#!/usr/bin/env python3

import uproot4 as uproot
import awkward1 as ak
import numpy as np
import numba

from python.Process import Process

class Jet(Process):
    def __init__(self, process):
        super().__init__(process)
        self.extraFuncs = [("jet_mask", "Jet_jetMask", None, Jet.jet),
                           ("bjet_mask", "Jet_bjetMask", None, Jet.bjet),
                           ("calc_HT", "Jet_HT", "Jet_jetMask", Jet.ht)]
        
    # Numba methods
    jet = Process.prefix("Jet", ["pt", "eta", "jetId"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,i4)')
    def jet_mask(pt, eta, jetId):
        jetId_key = 0b11
        return (
            pt > 40 and
            np.abs(eta) < 2.4 and
            (jetId & jetId_key) != 0
        )
                                                            
    bjet = Process.prefix("Jet", ["pt", "eta", "jetId", "btagDeepB"])
    @staticmethod
    @numba.vectorize('b1(f4,f4,i4,f4)')
    def bjet_mask(pt, eta, jetId, btag):
        jetId_key = 0b11
        return (
            pt > 25 and
            np.abs(eta) < 2.4 and
            (jetId & jetId_key) != 0 and
            btag > 0.6324
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

# size_t ThreeLepSelector::getCloseJetIndex(LorentzVector& lep, double minDR ) {
#     size_t minIndex = -1;

#     for(size_t index = 0; index < nJet; ++index) {
#         LorentzVector jet = get4Vector(PID_JET, index);
#         double dr = reco::deltaR(jet, lep);
#         if(minDR > dr) {
#             minDR = dr;
#             minIndex = index;
#         }
#     }
#     return minIndex;
# }
#     @numba.jit(nopython=True)
#     def calc_HT(events, builder):
#         for event in events:
#             HT = 0
#             for j in range(len(event["Jet_pt"])):
#                 HT += event.Jet_pt[j]
#             builder.real(HT)


# bool ThreeLepSelector::passFullIso(LorentzVector& lep, double I2, double I3) {
#     int closeIdx = getCloseJetIndex(lep);
#     LorentzVector closeJet  = get4Vector(PID_JET, closeIdx);
# #ifdef CLOSEJET_REWEIGHT
#     closeJet = (Jet_L1[closeIdx]*(1-Jet_rawFactor[closeIdx])*closeJet-lep)*Jet_L2L3[closeIdx]+lep;
# #endif // CLOSEJET_REWEIGHT

#     return (lep.Pt()/closeJet.Pt() > I2) || (LepRelPt(lep, closeJet) > I3);
# }


# double ThreeLepSelector::LepRelPt(LorentzVector& lep, LorentzVector& closeJet) {
#     auto diff = closeJet.Vect() - lep.Vect();
#     auto cross = diff.Cross(lep.Vect());
#     return std::sqrt(cross.Mag2()/diff.Mag2());
# }



# bool ThreeLepSelector::passFakeableCuts(GoodPart& lep) {
#     int index = lep.index;
#     if(lep.Pt() < 10) return false;
#     if(lep.Id() == PID_MUON) {
#         return (Muon_mediumId[index]
#            && Muon_sip3d[index] < 4
#            && Muon_tightCharge[index] == 2
#            && passFullIso(lep.v, 0.72, 7.2)

# 		);
#     }
#     else {
#         return (Electron_sip3d[index] < 4
#            && Electron_tightCharge[index] == 2
#            && Electron_lostHits[index] == 0
#            && passFullIso(lep.v, 0.8, 7.2)
# 		);
#     }
# }
