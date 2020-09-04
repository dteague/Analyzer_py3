from .ThreeTop_Muon import Muon
from .ThreeTop_Jets import Jet
from .ThreeTop_Electron import Electron
from .ThreeTop_Event import EventWide

from python.Scheduler import Scheduler
from python.CutApplier import CutApplier
from Utilities.FileGetter import pre

# Run Specifics
Scheduler.add_step([Muon, Electron])
Scheduler.add_step([Jet])
Scheduler.add_step([EventWide])

# Applying specifics
CutApplier.add_scale_factor("Event_wDecayScale")
CutApplier.add_scale_factor("Event_pileupScale")

CutApplier.add_cut("Event_MetFilterMask")
# CutApplier.add_cut("Event_triggerMask")
CutApplier.add_cut("Event_HT > 150")
CutApplier.add_cut("(Event_channels) > 1")
CutApplier.add_cut("abs(Event_channels) < 30 ")
CutApplier.add_cut("ak.count_nonzero(Jet_jetMask, axis=1) >= 2")
CutApplier.add_cut("ak.count_nonzero(Jet_bjetMask, axis=1) >= 1")
CutApplier.add_cut("Muon_ZVeto")
CutApplier.add_cut("Electron_ZVeto")


CutApplier.add_vars("looseMuon", pre("Muon", ["pt", "eta", "phi", "mass"]), "Muon_fakeMask")
CutApplier.add_vars("tightMuon", pre("Muon", ["pt", "eta", "phi", "mass"]), "Muon_finalMask")
CutApplier.add_vars("looseElectron", pre("Electron", ["pt", "eCorr", "eta", "phi", "mass"]), "Electron_fakeMask")
CutApplier.add_vars("tightElectron", pre("Electron", ["pt", "eCorr", "eta", "phi", "mass"]), "Electron_finalMask")
CutApplier.add_vars("Jets", pre("Jet", ["pt", "eta", "phi", "mass"]), "Jet_jetMask")
CutApplier.add_vars("BJets", pre("Jet", ["pt", "eta", "phi", "mass"]), "Jet_bjetMask")


CutApplier.add_vars("Event_MET", ["MET_pt", "MET_phi"])

CutApplier.add_vars_derived(
    "Event_variables", pre("Event", ["HT", "channels", "centrality", "sphericity"]))
