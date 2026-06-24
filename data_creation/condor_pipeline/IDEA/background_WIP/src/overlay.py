#
# Copyright (c) 2020-2024 Key4hep-Project.
#
# This file is part of Key4hep.
# See https://key4hep.github.io/key4hep-doc/ for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from Gaudi.Configuration import INFO

from k4FWCore import ApplicationMgr
from k4FWCore import IOSvc
from Configurables import EventDataSvc
from Configurables import OverlayTiming
from Configurables import UniqueIDGenSvc

from pathlib import Path


background_base = Path("/eos/experiment/fcc/ee/simulation/key4hep_2026_04_20/91GeV/IDEA_o1_v03/IPC_Z_background")
background_file_list = []
for d in sorted(background_base.iterdir()):
    background_file_list.append(str(d))
            
id_service = UniqueIDGenSvc("UniqueIDGenSvc")
eds = EventDataSvc("EventDataSvc")
iosvc = IOSvc()
# iosvc.Input = "/afs/cern.ch/user/a/aloeschc/fcc_fullsim_testing_grounds/data/IDEA/IDEA_o1_v03/physics_events/p8_ee_Z_qqbar_ud/1000_91.188GeV_ISR_FSR/000/IDEA_o1_v03_1000_p8_ee_Z_qqbar_ud_91.188GeV_ISR_FSR.root"
iosvc.Input = "/afs/cern.ch/work/a/adevita/public/testBIB/bib-studies/outSim_399.root" 

iosvc.Output = "IDEA_o1_v03_OverlayIPP_1000_p8_ee_Z_qqbar_ud_91.root"

overlay = OverlayTiming()
overlay.MCParticles = "MCParticles"
overlay.BackgroundMCParticleCollectionName = "MCParticles"
overlay.SimTrackerHits = ["DCHCollection", "MuonSystemCollection", "SiWrDCollection", "SiWrBCollection", "VertexBarrelCollection", "VertexEndcapCollection", "PreshowerSystemCollection"]
overlay.SimCalorimeterHits = []
overlay.OutputSimTrackerHits = ["OverlayDCHCollection", "OverlayMuonSystemCollection", "OverlaySiWrDCollection", "OverlaySiWrBCollection", "OverlayVertexBarrelCollection", "OverlayVertexEndcapCollection", "OverlayPreshowerSystemCollection"]
overlay.OutputSimCalorimeterHits = []
overlay.OutputMCParticles = "OverlayMCParticles"
overlay.OutputCaloHitContributions = []
# overlay.StartBackgroundEventIndex = 0
overlay.AllowReusingBackgroundFiles = True
overlay.CopyCellIDMetadata = True
overlay.NBunchtrain = 41          # total BX in train
overlay.NumberBackground = [1]    # one background event per BX
overlay.Delta_t = 20              # ns between BX
overlay.PhysicsBX = 21            # puts physics at 21 with 20 before & 30 after (allow for hits 200ns after event time)
overlay.Poisson_random_NOverlay = [False]
overlay.StartBackgroundEventIndex = -1
overlay.BackgroundFileNames = [
      background_file_list
]
overlay.TimeWindows = {"MCParticles": [-400, 400], "DCHCollection": [-400, 400], "MuonSystemCollection": [-20, 0], "SiWrDCollection": [-20, 0],"SiWrBCollection": [-20, 0], "VertexBarrelCollection": [-20, 0],"VertexEndcapCollection": [-20, 0], "PreshowerSystemCollection": [-20, 0]}

iosvc.outputCommands = ["drop *", "keep OverlayDCHCollection*", "keep OverlaySiWrDCollection*", "keep OverlaySiWrBCollection*", "keep OverlayVertexBarrelCollection*", "keep OverlayVertexEndcapCollection*", "keep OverlayMC*", "keep *EventHeader*"]


ApplicationMgr(TopAlg=[overlay],
               EvtSel="NONE",
               EvtMax=1,
               ExtSvc=[eds],
               OutputLevel=INFO,
               )