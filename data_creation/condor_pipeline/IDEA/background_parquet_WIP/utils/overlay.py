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
from k4FWCore.parseArgs import parser
from Configurables import EventDataSvc
from Configurables import OverlayTiming
from Configurables import UniqueIDGenSvc

from pathlib import Path


parser.add_argument("--inputFile", required=True, help="Physics simulation input")
parser.add_argument("--outputFile", required=True, help="Overlaid EDM4hep output")
parser.add_argument(
    "--backgroundFilesPath",
    required=True,
    help="Directory containing the background ROOT files",
)
parser.add_argument(
    "--numEvents",
    type=int,
    default=-1,
    help="Number of physics events to overlay; -1 processes all events",
)
args = parser.parse_args()

background_base = Path(args.backgroundFilesPath).expanduser().resolve()
if not background_base.is_dir():
    raise RuntimeError(f"Background directory not found: {background_base}")

background_file_list = sorted(
    str(path.resolve())
    for path in background_base.glob("*.root")
    if path.is_file()
)
if not background_file_list:
    raise RuntimeError(f"No .root background files found directly in {background_base}")

id_service = UniqueIDGenSvc("UniqueIDGenSvc")
eds = EventDataSvc("EventDataSvc")
iosvc = IOSvc()
iosvc.Input = str(Path(args.inputFile).expanduser().resolve())
iosvc.Output = str(Path(args.outputFile).expanduser().resolve())

overlay = OverlayTiming()

overlay.MCParticles = "MCParticles"
overlay.BackgroundMCParticleCollectionName = "MCParticles"
overlay.OutputMCParticles = "OverlayMCParticles"

overlay.SimTrackerHits = ["DCHCollection", "MuonSystemCollection", "SiWrDCollection", "SiWrBCollection", "VertexBarrelCollection", "VertexEndcapCollection", "PreshowerSystemCollection"]
overlay.OutputSimTrackerHits = ["OverlayDCHCollection", "OverlayMuonSystemCollection", "OverlaySiWrDCollection", "OverlaySiWrBCollection", "OverlayVertexBarrelCollection", "OverlayVertexEndcapCollection", "OverlayPreshowerSystemCollection"]

overlay.SimCalorimeterHits = []
overlay.OutputSimCalorimeterHits = []
overlay.OutputCaloHitContributions = []

# overlay.StartBackgroundEventIndex = 0
overlay.AllowReusingBackgroundFiles = True
overlay.CopyCellIDMetadata = True
overlay.NBunchtrain = 33                                # 16 before, signal, 16 after
overlay.NumberBackground = [1]                          # one background event per BX
overlay.Delta_t = 25                                    # ns between BX
overlay.PhysicsBX = 17                                  # puts physics at BX 17, with offsets from -16 to +16 BX
overlay.Poisson_random_NOverlay = [False]
overlay.StartBackgroundEventIndex = -1
overlay.BackgroundFileNames = [background_file_list]
overlay.RandomMixBackgroundFiles = True

overlay.TimeWindows = {"MCParticles": [-400, 400], "DCHCollection": [-400, 400], "MuonSystemCollection": [0, 25.], "SiWrDCollection": [0, 25.],"SiWrBCollection": [0, 25.], "VertexBarrelCollection": [0, 25.],"VertexEndcapCollection": [0, 25.], "PreshowerSystemCollection": [0, 25.]}

iosvc.outputCommands = ["drop *", "keep OverlayDCHCollection*", "keep OverlaySiWrDCollection*", "keep OverlaySiWrBCollection*", "keep OverlayVertexBarrelCollection*", "keep OverlayVertexEndcapCollection*", "keep OverlayMC*", "keep *EventHeader*"]


ApplicationMgr(
    TopAlg=[overlay],
    EvtSel="NONE",
    EvtMax=args.numEvents,
    ExtSvc=[eds, id_service],
    OutputLevel=INFO,
)
