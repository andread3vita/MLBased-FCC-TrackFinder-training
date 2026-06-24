from podio import root_io
import sys
import ROOT
from ROOT import TFile, TTree

from tools_tree import (
    read_mc_collection,
    clear_dic,
    initialize,
    store_hit_col_SenseWireHits,
    store_hit_col_PlanarHits,
    merge_list_MCS,
    gen_particles_find,
)

debug = False
rootfile = sys.argv[1]

reader = root_io.Reader(rootfile)
output_file = sys.argv[2]

det_version = int(sys.argv[3])
det_option = int(sys.argv[4])

metadata = reader.get("metadata")[0]

out_root = TFile(output_file, "RECREATE")
out_root.cd()    
t = TTree("events", "tracking tree")
event_number, n_hit, n_part, dic, t = initialize(t)

event_number[0] = 0
event_numbers = 0
i = 0
for event in reader.get("events"):
    
    print("Analysing event:",event_number[0])
    
    (
        genpart_indexes_pre,
        indexes_genpart_pre,
        n_part_pre,
        total_e,
        e_pp
    ) = gen_particles_find(event, debug)
    
    clear_dic(dic)
    n_part[0] = 0
    
    if (det_version == 3 and det_option == 1):
        
        n_hit, dic, list_of_MCs1 = store_hit_col_SenseWireHits(
            event,
            n_hit,
            dic,
            metadata
        )
            
    else:
        print("Sense Wire Hit analyser not yet implemented!")
    
    n_hit, dic, list_of_MCs2 = store_hit_col_PlanarHits(
        event,
        n_hit,
        dic
    )
    
    unique_MCS = merge_list_MCS(list_of_MCs1, list_of_MCs2)
    n_part, dic = read_mc_collection(event, dic, n_part, debug, unique_MCS)
    event_number[0] += 1
    t.Fill()


t.Write("", ROOT.TObject.kOverwrite)
out_root.Write("", ROOT.TObject.kOverwrite)
out_root.Close()
