from podio import root_io
import edm4hep
import ROOT
from ROOT import TFile, TTree
from array import array
import math
import dd4hep as dd4hepModule
from ROOT import dd4hep
import numpy as np

def get_genparticle_daughters(i, mcparts):

    p = mcparts[i]
    daughters = p.getDaughters()
    daughter_positions = []

    for daughter in daughters:
        daughter_positions.append(daughter.getObjectID().index)

    return daughter_positions

def get_genparticle_parents(i, mcparts):

    p = mcparts[i]
    parents = p.getParents()
    parent_positions = []

    for parent in parents:
        parent_positions.append(parent.getObjectID().index)

    return parent_positions

def gen_particles_find(event, debug):

    genparts = "MCParticles"
    gen_part_coll = event.get(genparts)
    genpart_indexes_pre = (
        dict()
    )  ## key: index in gen particle collection, value: position in stored gen particle array
    indexes_genpart_pre = (
        dict()
    )  ## key: position in stored gen particle array, value: index in gen particle collection
    
    total_e = 0
    n_part_pre = 0
    e_pp = np.zeros(11)
    for j, part in enumerate(gen_part_coll):
        momentum = part.getMomentum()
        p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
        if debug:
            if j < 11 and j > 1:
                e_pp[j] = p
                total_e = total_e + p
        theta = math.acos(momentum.z / p)
        phi = math.atan2(momentum.y, momentum.x)

        if debug:
            if part.getGeneratorStatus() == 1:
                print(
                    "all genparts: N: {}, PID: {}, Q: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, M: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, status: {}, parents: {}, daughters: {}, decayed_traacker: {}".format(
                        j,
                        part.getPDG(),
                        part.getCharge(),
                        p,
                        theta,
                        phi,
                        part.getMass(),
                        part.getVertex().x * 1e-03,
                        part.getVertex().y * 1e-03,
                        math.sqrt(part.getVertex().x ** 2 + part.getVertex().y ** 2)
                        * 1e-03,
                        part.getVertex().z * 1e-03,
                        part.getGeneratorStatus(),
                        get_genparticle_parents(
                            j,
                            gen_part_coll,
                        ),
                        get_genparticle_daughters(
                            j,
                            gen_part_coll,
                        ),
                        part.isDecayedInTracker() * 1,
                    )
                )

        ## store all gen parts for now
        genpart_indexes_pre[j] = n_part_pre
        indexes_genpart_pre[n_part_pre] = j
        n_part_pre += 1

        """
        # exclude neutrinos (and pi0 for now)
        if part.generatorStatus == 1 and abs(part.PDG) not in [12, 14, 16, 111]:

            genpart_indexes_pre[j] = n_part_pre
            indexes_genpart_pre[n_part_pre] = j
            n_part_pre += 1

        # extract the photons from the pi0
        elif part.generatorStatus == 1 and part.PDG == 111:

            daughters = get_genparticle_daughters(
                j, gen_part_coll, gen_daughter_link_indexmc
            )

            if len(daughters) != 2:
                print("STRANGE PI0 DECAY")

            for d in daughters:
                a = gen_part_coll[d]
                genpart_indexes_pre[d] = n_part_pre
                indexes_genpart_pre[n_part_pre] = d
                n_part_pre += 1
        """
    return (
        genpart_indexes_pre,
        indexes_genpart_pre,
        n_part_pre,
        total_e,
        e_pp
    )


def find_mother_particle(mc_particle):
    parent_p = mc_particle.getObjectID().index
    counter = 0
    decayed_in_tracker = 1
    while decayed_in_tracker == 1:
        if type(parent_p) == list:
            parent_p = parent_p[0]
        parents = mc_particle.getParents()
        parent_p_r = []
        for parent in parents:
            parent_p_r.append(parent.getObjectID().index)
            decayed_in_tracker = parent.isDecayedInTracker() * 1
        pp_old = parent_p
        counter = counter + 1
        parent_p = parent_p_r
        if len(np.reshape(np.array(parent_p), -1)) > 1.5:
            parent_p = pp_old
            decayed_in_tracker = 0
    return parent_p


def initialize(t):
    
    event_number = array("i", [0])
    n_hit = array("i", [0])
    n_part = array("i", [0])

    hit_EDep = ROOT.std.vector("float")()
    hit_time = ROOT.std.vector("float")()

    # for true hits
    hit_pathLength = ROOT.std.vector("float")()
    hit_x_true = ROOT.std.vector("float")()
    hit_y_true = ROOT.std.vector("float")()
    hit_z_true = ROOT.std.vector("float")()
    hit_px = ROOT.std.vector("float")()
    hit_py = ROOT.std.vector("float")()
    hit_pz = ROOT.std.vector("float")()
    
    # for digitized hits
    hit_x = ROOT.std.vector("float")()
    hit_y = ROOT.std.vector("float")()
    hit_z = ROOT.std.vector("float")()
    leftPosition_x = ROOT.std.vector("float")()
    leftPosition_y = ROOT.std.vector("float")()
    leftPosition_z = ROOT.std.vector("float")()
    rightPosition_x = ROOT.std.vector("float")()
    rightPosition_y = ROOT.std.vector("float")()
    rightPosition_z = ROOT.std.vector("float")()
    cluster_count = ROOT.std.vector("float")()
    
    # particle info
    hit_type = ROOT.std.vector("float")()
    hit_particle_index = ROOT.std.vector("float")()
    part_p = ROOT.std.vector("float")()
    part_p_t = ROOT.std.vector("float")()
    part_theta = ROOT.std.vector("float")()
    part_phi = ROOT.std.vector("float")()
    part_m = ROOT.std.vector("float")()
    part_id = ROOT.std.vector("float")()
    part_parent = ROOT.std.vector("float")()
    part_pid = ROOT.std.vector("float")()

    # cellID and other hit info
    hit_cellID = ROOT.std.vector("int")()
    superLayer = ROOT.std.vector("float")()
    layer = ROOT.std.vector("float")()
    phi = ROOT.std.vector("float")()
    stereo = ROOT.std.vector("float")()
    gen_status = ROOT.std.vector("float")()
    
    produced_by_secondary = ROOT.std.vector("float")()

    t.Branch("event_number", event_number, "event_number/I")
    t.Branch("n_hit", n_hit, "n_hit/I")
    t.Branch("n_part", n_part, "n_part/I")
    t.Branch("gen_status", gen_status)
    t.Branch("hit_x_true", hit_x_true)
    t.Branch("hit_y_true", hit_y_true)
    t.Branch("hit_z_true", hit_z_true)
    t.Branch("hit_pathLength", hit_pathLength)
    t.Branch("hit_px", hit_px)
    t.Branch("hit_py", hit_py)
    t.Branch("hit_pz", hit_pz)


    t.Branch("hit_x", hit_x)
    t.Branch("hit_y", hit_y)
    t.Branch("hit_z", hit_z)
    t.Branch("leftPosition_x", leftPosition_x)
    t.Branch("leftPosition_y", leftPosition_y)
    t.Branch("leftPosition_z", leftPosition_z)
    t.Branch("rightPosition_x", rightPosition_x)
    t.Branch("rightPosition_y", rightPosition_y)
    t.Branch("rightPosition_z", rightPosition_z)
    t.Branch("cluster_count", cluster_count)
    t.Branch("hit_type", hit_type)
    t.Branch("hit_EDep", hit_EDep)
    t.Branch("hit_time", hit_time)
    t.Branch("hit_cellID", hit_cellID)

    t.Branch("hit_particle_index", hit_particle_index)
    t.Branch("part_p", part_p)
    t.Branch("part_p_t", part_p_t)
    t.Branch("part_theta", part_theta)
    t.Branch("part_phi", part_phi)
    t.Branch("part_m", part_m)
    t.Branch("part_pid", part_pid)
    t.Branch("part_id", part_id)
    t.Branch("superLayer", superLayer)
    t.Branch("layer", layer)
    t.Branch("phi", phi)
    t.Branch("stereo", stereo)
    t.Branch("part_parent", part_parent)
    t.Branch("produced_by_secondary", produced_by_secondary)

    dic = {
        "hit_x_true": hit_x_true,
        "hit_y_true": hit_y_true,
        "hit_z_true": hit_z_true,
        "hit_type": hit_type,
        "hit_EDep": hit_EDep,
        "hit_time": hit_time,
        "hit_pathLength": hit_pathLength,
        "hit_particle_index": hit_particle_index,
        "hit_px": hit_px,
        "hit_py": hit_py,
        "hit_pz": hit_pz,
        "part_p": part_p,
        "part_p_t": part_p_t,
        "part_theta": part_theta,
        "part_phi": part_phi,
        "part_m": part_m,
        "part_pid": part_pid,
        "part_id": part_id,
        "gen_status": gen_status,
        "hit_cellID": hit_cellID,
        "hit_x": hit_x,
        "hit_y": hit_y,
        "hit_z": hit_z,
        "leftPosition_x": leftPosition_x,
        "leftPosition_y": leftPosition_y,
        "leftPosition_z": leftPosition_z,
        "rightPosition_x": rightPosition_x,
        "rightPosition_y": rightPosition_y,
        "rightPosition_z": rightPosition_z,
        "produced_by_secondary": produced_by_secondary,
        "cluster_count": cluster_count,
        "superLayer": superLayer,
        "layer": layer,
        "phi": phi,
        "stereo": stereo,
        "part_parent": part_parent,
    }
    
    return (event_number, n_hit, n_part, dic, t)


def read_mc_collection(event, dic, n_part, debug, unique_MCS):
    mc_particles = event.get("MCParticles")
    for jj, mc_particle in enumerate(mc_particles):

        pdg = mc_particle.getPDG()
        m = mc_particle.getMass()
        p_ = mc_particle.getMomentum()
        p = math.sqrt(p_.x**2 + p_.y**2 + p_.z**2)
        p_t = math.sqrt(p_.x**2 + p_.y**2)
        object_id_particle = mc_particle.getObjectID()
        genlink0_particle = object_id_particle.index
        # only store particles that have hits
        if np.sum(unique_MCS == jj) > 0:
            if p > 0:
                theta = math.acos(p_.z / p)
                phi = math.atan2(p_.y, p_.x)
                dic["part_p"].push_back(p)
                dic["part_p_t"].push_back(p_t)
                dic["part_theta"].push_back(theta)
                dic["part_phi"].push_back(phi)
                dic["part_m"].push_back(m)
                dic["part_pid"].push_back(pdg)
                dic["part_id"].push_back(genlink0_particle)
                
            else:
                theta = 0.0
                phi = 0.0
                dic["part_p"].push_back(p)
                dic["part_p_t"].push_back(p_t)
                dic["part_theta"].push_back(theta)
                dic["part_phi"].push_back(phi)
                dic["part_m"].push_back(m)
                dic["part_pid"].push_back(pdg)
                dic["part_id"].push_back(genlink0_particle)
                
            parents = mc_particle.getParents()
            dic["part_parent"].push_back(parents[0].getObjectID().index)
            dic["gen_status"].push_back(mc_particle.getGeneratorStatus())
        if debug and  np.sum(unique_MCS == jj) > 0:
            if mc_particle.getGeneratorStatus() ==1:
                print("gen status 1 part")
                print(
                    "all genparts: N: {}, PID: {}, Q: {}, P: {:.2e}, status: {}, parents: {}, daughters: {}, decayed_traacker: {}".format(
                        jj,
                        mc_particle.getPDG(),
                        mc_particle.getCharge(),
                        p,

                        mc_particle.getGeneratorStatus(),
                        get_genparticle_parents(
                            genlink0_particle,
                            mc_particles,
                        ),
                        get_genparticle_daughters(
                            genlink0_particle,
                            mc_particles,
                        ),
                        mc_particle.isDecayedInTracker() * 1,
                    )
                )
        if np.sum(unique_MCS == jj) > 0:
            n_part[0] += 1

    return n_part, dic


def clear_dic(dic):
    for key in dic:
        dic[key].clear()
    return dic

def local_to_global(local_pos, x_prime, y_prime, z_prime, wire_pos):
    """
    Transforms local coordinates to global coordinates using the provided basis vectors.

    Parameters:
        local_pos (np.array): Local position vector [x, y, z].
        x_prime (np.array): x' axis unit vector.
        y_prime (np.array): y' axis unit vector.
        z_prime (np.array): z' axis unit vector.
        wire_pos (np.array): Wire position in global coordinates.

    Returns:
        np.array: Global position vector.
    """
    global_pos = x_prime * local_pos[0] + y_prime * local_pos[1] + z_prime * local_pos[2] + wire_pos
    return global_pos

def store_hit_col_SenseWireHits(
    event,
    n_hit,
    dic,
    metadata
):
    
    hit_links = event.get("DCH_DigiSimAssociationCollection")
    digi_hit_collection = event.get("DCH_DigiCollection")
    
    n_hit[0] = 0
    list_of_MC = []
    
    for idx_link, link in enumerate(hit_links):
        
        sim_hit = link.getTo()
        digi_hit = digi_hit_collection[idx_link]
        
        cellID = sim_hit.getCellID()
        EDep = sim_hit.getEDep()
        time = sim_hit.getTime()
        
        #  position along the wire, wire direction and drift distance
        wirePos = digi_hit.getPosition()
        wirePos = np.array([wirePos[0],wirePos[1],wirePos[2]])
    
        distanceToWire = digi_hit.getDistanceToWire()
        wire_azimuthal_angle = digi_hit.getWireAzimuthalAngle()
        wire_stereo_angle = digi_hit.getWireStereoAngle()
        
        d_x = np.sin(wire_stereo_angle) * np.sin(wire_azimuthal_angle)
        d_y = -(np.sin(wire_stereo_angle) * np.cos(wire_azimuthal_angle))
        d_z = np.cos(wire_stereo_angle)    

        # z_prime
        z_prime = np.array([d_x, d_y, d_z])
        norm_z_prime = np.linalg.norm(z_prime)
        z_prime /= norm_z_prime

        # x_prime
        x_prime = np.array([1.0, 0.0, -d_x / d_z])
        norm_x_prime = np.linalg.norm(x_prime)
        x_prime /= norm_x_prime

        # y_prime (cross product)
        y_prime = np.cross(z_prime, x_prime)
        norm_y_prime = np.linalg.norm(y_prime)
        y_prime /= norm_y_prime

        # Conversion from local to global
        left_hit_local_position = np.array([-distanceToWire, 0.0, 0.0])
        right_hit_local_position = np.array([distanceToWire, 0.0, 0.0])
            
        left_hit_global_position = local_to_global(left_hit_local_position, x_prime, y_prime, z_prime, wirePos)
        right_hit_global_position = local_to_global(right_hit_local_position, x_prime, y_prime, z_prime, wirePos)
        
        # Check d_z
        if abs(d_z) < 1e-12 or np.isnan(d_z) or np.isinf(d_z):
            raise ValueError(f"d_z invalid! value={d_z}, wire_stereo_angle={wire_stereo_angle}")

        # norm_z_prime
        if norm_z_prime < 1e-12 or np.isnan(norm_z_prime) or np.isinf(norm_z_prime):
            raise ValueError(f"z_prime norm invalid! norm_z_prime={norm_z_prime}")

        # norm_x_prime
        if norm_x_prime < 1e-12 or np.isnan(norm_x_prime) or np.isinf(norm_x_prime):
            raise ValueError(f"x_prime norm invalid! norm_x_prime={norm_x_prime}")

        # norm_y_prime
        if norm_y_prime < 1e-12 or np.isnan(norm_y_prime) or np.isinf(norm_y_prime):
            raise ValueError(f"y_prime norm invalid! norm_y_prime={norm_y_prime}")

        cluster_count = digi_hit.getNClusters()

        dic["hit_x"].push_back(wirePos[0])
        dic["hit_y"].push_back(wirePos[1])
        dic["hit_z"].push_back(wirePos[2])
        dic["leftPosition_x"].push_back(left_hit_global_position[0])
        dic["leftPosition_y"].push_back(left_hit_global_position[1])
        dic["leftPosition_z"].push_back(left_hit_global_position[2])
        dic["rightPosition_x"].push_back(right_hit_global_position[0])
        dic["rightPosition_y"].push_back(right_hit_global_position[1])
        dic["rightPosition_z"].push_back(right_hit_global_position[2])

        dic["cluster_count"].push_back(cluster_count)

        pathLength = sim_hit.getPathLength()
        position = sim_hit.getPosition()
        x = position.x
        y = position.y
        z = position.z
        momentum = sim_hit.getMomentum()
        px = momentum.x
        py = momentum.y
        pz = momentum.z
        produced_by_secondary = sim_hit.isProducedBySecondary()
        dic["hit_x_true"].push_back(x)
        dic["hit_y_true"].push_back(y)
        dic["hit_z_true"].push_back(z)
        p = math.sqrt(px * px + py * py + pz * pz)
        dic["hit_px"].push_back(px)
        dic["hit_py"].push_back(py)
        dic["hit_pz"].push_back(pz)

        dic["produced_by_secondary"].push_back(1.0 * produced_by_secondary)

        htype = 0

        # dummy example, cellid_encoding = "foo:2,bar:3,baz:-4"
        cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
        decoder = dd4hep.BitFieldCoder(cellid_encoding)
        superLayer = decoder.get(cellID, "superlayer")
        layer = decoder.get(cellID, "layer")
        phi = decoder.get(cellID, "nphi")
        stereo = decoder.get(cellID, "stereosign")
        dic["hit_cellID"].push_back(cellID)
        dic["hit_EDep"].push_back(EDep)
        dic["hit_time"].push_back(time)
        dic["hit_pathLength"].push_back(pathLength)
        dic["hit_type"].push_back(htype)
        dic["superLayer"].push_back(superLayer)
        dic["layer"].push_back(layer)
        dic["phi"].push_back(phi)
        dic["stereo"].push_back(stereo)

        mcParticle = sim_hit.getParticle()
        object_id = mcParticle.getObjectID()
        hit_particle_index = object_id.index
        dic["hit_particle_index"].push_back(hit_particle_index)
        list_of_MC.append(hit_particle_index)
        
        n_hit[0] += 1
        
    return n_hit, dic, list_of_MC         

def store_hit_col_PlanarHits(
    event,
    n_hit,
    dic
):
    
    vtxD_links = event.get("VTXDSimDigiLinks")
    vtxB_links = event.get("VTXBSimDigiLinks")
    Siw_D_links = event.get("SiWrDSimDigiLinks")
    Siw_B_links = event.get("SiWrBSimDigiLinks")
    
    hit_collections_links = [vtxD_links, vtxB_links,Siw_D_links,Siw_B_links]
    
    list_of_MC = []
    for coll in hit_collections_links:

        for link in coll:
            
            hit_digi = link.getFrom()
            hit_sim = link.getTo()
            
            EDep = hit_digi.getEDep()
            time = hit_digi.getTime()
            cellID = hit_sim.getCellID()

            # digi hit
            position_digi = hit_digi.getPosition()

            dic["hit_x"].push_back(position_digi.x)
            dic["hit_y"].push_back(position_digi.y)
            dic["hit_z"].push_back(position_digi.z)
            dic["leftPosition_x"].push_back(0)
            dic["leftPosition_y"].push_back(0)
            dic["leftPosition_z"].push_back(0)
            dic["rightPosition_x"].push_back(0)
            dic["rightPosition_y"].push_back(0)
            dic["rightPosition_z"].push_back(0)
            dic["cluster_count"].push_back(0)


            # sim hit

            position_sim = hit_sim.getPosition()
            pathLength = hit_sim.getPathLength()
            momentum = hit_sim.getMomentum()
            produced_by_secondary = hit_sim.isProducedBySecondary()

            px = momentum.x
            py = momentum.y
            pz = momentum.z

            htype = 1
           
            dic["produced_by_secondary"].push_back(1.0 * produced_by_secondary)
            dic["hit_cellID"].push_back(cellID)
            dic["hit_EDep"].push_back(EDep)
            dic["hit_time"].push_back(time)
            dic["hit_pathLength"].push_back(pathLength)

            dic["hit_x_true"].push_back(position_sim.x)
            dic["hit_y_true"].push_back(position_sim.y)
            dic["hit_z_true"].push_back(position_sim.z)
            
            dic["hit_px"].push_back(px)
            dic["hit_py"].push_back(py)
            dic["hit_pz"].push_back(pz)
            
            dic["hit_type"].push_back(htype)
        
            dic["superLayer"].push_back(0)
            dic["layer"].push_back(0)
            dic["phi"].push_back(0)
            dic["stereo"].push_back(0)

            
            mcParticle = hit_sim.getParticle()
            object_id = mcParticle.getObjectID()
            hit_particle_index = object_id.index

            dic["hit_particle_index"].push_back(hit_particle_index)
            list_of_MC.append(hit_particle_index)
            n_hit[0] += 1

    return n_hit, dic, list_of_MC


def merge_list_MCS(list_1, list_2):
    list_1 = np.array(list_1)
    list_2 = np.array(list_2)
    unique_m1 = np.unique(list_1)
    unique_m2 = np.unique(list_2)
    unique_mc = np.concatenate((unique_m1, unique_m2), axis=0)

    return unique_mc
