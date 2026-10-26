Main:numberOfEvents = 100          ! number of events to generate
Main:timesAllowErrors = 5          ! how many aborts before run stops

! Settings related to output in init(), next() and stat().
Init:showChangedSettings = on
Init:showChangedParticleData = off
Next:numberCount = 100
Next:numberShowInfo = 1
Next:numberShowProcess = 1
Next:numberShowEvent = 0
Stat:showPartonLevel = off

! e+e- collisions at the Z pole.
Beams:idA = 11
Beams:idB = -11
Beams:eCM = 91
Beams:allowMomentumSpread = off

! Vertex smearing is applied by the GaussSmearVertex tool in pythia.py:
! sigmaX = 5.96 um, sigmaY = 23.8 nm, sigmaZ = 0.397 mm.

PartonLevel:ISR = off
PartonLevel:FSR = off

! Generate Z/gamma* and allow only the b bbar decay channel.
WeakSingleBoson:ffbar2gmZ = on
23:onMode = off
23:onIfAny = 5

ParticleDecays:limitCylinder = on
ParticleDecays:xyMax = 2250.
ParticleDecays:zMax = 2500.
