from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
import dgl
from scipy.spatial import distance_matrix
import torch
from dgllife.utils import BaseAtomFeaturizer,BaseBondFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot
from functools import partial
import os
import torch
import dgl
from dgl import save_graphs, load_graphs
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import math
from get_pdbbind_utils import log,load_molecule,multiprocess_load_molecules
import warnings
warnings.filterwarnings('ignore')




#############################################################################################
##                      获得PDBBIND的数据文件的路径以及转换成为MOL文件                            ##
##          Obtain the path of PDBBIND data file and convert it into MOL file              ##
#############################################################################################

class pdbbind(Dataset):
     def __init__(self,root_dir_path,index_file,num_processes=1,sanitize=False, calc_charges=False,
                                remove_hs=True, use_conformation=True,only_polar_hydrogens=False,use_pocket=True,get_complex_mol=False):
         self.root_dir_path = root_dir_path
         self.index_file = index_file
         self.num_processes = num_processes
         self.sanitize = sanitize
         self.calc_charge = calc_charges
         self.remove_hs = remove_hs
         self.use_conformation = use_conformation
         self.only_polar_hydrogens = only_polar_hydrogens
         self.use_pocket = use_pocket
         self.get_complex_mol = get_complex_mol

     def getpKd(self,binding_data):
         labels = []
         for line in binding_data:
             unit = line[-2:]
             if unit == 'mM':
                 pKd = - math.log10(float(line[:-2]) * 1e-03)
                 # 保留两位小数
                 pKd = round(pKd, 2)

             elif unit == 'uM':
                 pKd = - math.log10(float(line[:-2]) * 1e-06)
                 # 保留两位小数
                 pKd = round(pKd, 2)

             elif unit == 'nM':
                 pKd = - math.log10(float(line[:-2]) * 1e-09)
                 pKd = round(pKd, 2)

             labels.append(pKd)
         return labels

     def files2mol(self):
         PDB_code = []
         binding_data = []
         with open(self.index_file, 'r') as f:
             for line in f.readlines():
                 if line[0] != "#":
                     splitted_elements = line.split()
                     PDB_code.append(splitted_elements[0])
                     splitted_pkd = splitted_elements[3]
                     data = splitted_pkd.split('=')[1]
                     binding_data.append(data)
         df = pd.DataFrame(PDB_code, columns=['PDB_code'])
         PDB_code = df['PDB_code'].tolist()

         ligand_files = [os.path.join(self.root_dir_path, "{}/".format(pdb), "{}_ligand.sdf".format(pdb)) for pdb in
                         PDB_code]

         if self.use_pocket:
             protein_files = [os.path.join(self.root_dir_path, "{}/".format(pdb), "{}_pocket.pdb".format(pdb)) for pdb
                              in PDB_code]
         else:
             protein_files = [os.path.join(self.root_dir_path, "{}/".format(pdb), "{}_protein.pdb".format(pdb)) for pdb
                              in PDB_code]



         pKd = self.getpKd(binding_data=binding_data)

         if self.get_complex_mol == True:
             if self.num_processes == 1:
                 log(f'Loading {len(PDB_code)} complexes.')
                 ligands_loaded = []
                 lig_to_remove = []
                 for pdb in tqdm(ligand_files):
                     lig = load_molecule(pdb,
                                         sanitize=False,
                                         remove_hs=self.remove_hs)
                     if lig == None:
                         lig_to_remove.append(pdb)
                         continue
                     if self.only_polar_hydrogens:
                         for atom in lig.GetAtoms():
                             if atom.GetAtomicNum() == 1 and [x.GetAtomicNum() for x in atom.GetNeighbors()] == [6]:
                                 atom.SetAtomicNum(0)
                         lig = Chem.DeleteSubstructs(lig, Chem.MolFromSmarts('[#0]'))
                         Chem.SanitizeMol(lig)
                     ligands_loaded.append(lig)

                     log(f'Loading {len(PDB_code)} recpetors.')
                     recpetors_loaded = []
                     recpetors_to_remove = []
                     for pdb in tqdm(protein_files):
                         lig = load_molecule(pdb,
                                             sanitize=False,
                                             remove_hs=self.remove_hs)
                         if lig == None:
                             recpetors_to_remove.append(pdb)
                             continue
                         if self.only_polar_hydrogens:
                             for atom in lig.GetAtoms():
                                 if atom.GetAtomicNum() == 1 and [x.GetAtomicNum() for x in atom.GetNeighbors()] == [6]:
                                     atom.SetAtomicNum(0)
                             lig = Chem.DeleteSubstructs(lig, Chem.MolFromSmarts('[#0]'))
                             Chem.SanitizeMol(lig)
                         recpetors_loaded.append(lig)

             if self.num_processes > 1:
                 log(f'Multithreaded Loading {len(PDB_code)} complexes .')
                 ligands_loaded = multiprocess_load_molecules(ligand_files, sanitize=self.sanitize,
                                                              calc_charges=self.calc_charge,
                                                              remove_hs=self.remove_hs,
                                                              use_conformation=self.use_conformation,
                                                              num_processes=self.num_processes)
                 protein_loaded = multiprocess_load_molecules(protein_files, sanitize=self.sanitize,
                                                              calc_charges=self.calc_charge,
                                                              remove_hs=self.remove_hs,
                                                              use_conformation=self.use_conformation,
                                                              num_processes=self.num_processes)
         else:
             if self.num_processes == 1:
                 log(f'Loading {len(PDB_code)} complexes of ligand.')
                 ligands_loaded = []
                 lig_to_remove = []
                 for pdb in tqdm(ligand_files):
                     lig = load_molecule(pdb,
                                         sanitize=False,
                                         remove_hs=self.remove_hs)
                     if lig == None:
                         lig_to_remove.append(pdb)
                         continue
                     if self.only_polar_hydrogens:
                         for atom in lig.GetAtoms():
                             if atom.GetAtomicNum() == 1 and [x.GetAtomicNum() for x in atom.GetNeighbors()] == [6]:
                                 atom.SetAtomicNum(0)
                         lig = Chem.DeleteSubstructs(lig, Chem.MolFromSmarts('[#0]'))
                         Chem.SanitizeMol(lig)
                     ligands_loaded.append(lig)

                 log(f'Loading {len(PDB_code)} complexes of protein.')
                 protein_loaded = []
                 pro_to_remove = []
                 for pdb in tqdm(protein_files):
                     pro = load_molecule(pdb,
                                         sanitize=False,
                                         remove_hs=self.remove_hs)
                     if pro == None:
                         pro_to_remove.append(pdb)
                         continue
                     if self.only_polar_hydrogens:
                         for atom in pro.GetAtoms():
                             if atom.GetAtomicNum() == 1 and [x.GetAtomicNum() for x in atom.GetNeighbors()] == [6]:
                                 atom.SetAtomicNum(0)
                         pro = Chem.DeleteSubstructs(pro, Chem.MolFromSmarts('[#0]'))
                         Chem.SanitizeMol(pro)
                     protein_loaded.append(pro)




         # return tuple(mol,3d_coord)

         return PDB_code,pKd,ligand_files,protein_files ,ligands_loaded,protein_loaded




def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})



def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['C'][nodes_ls[0]].numpy(), g.ndata['C'][nodes_ls[1]].numpy(),
                                            g.ndata['C'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()

def graphs_from_mol_v1(m1, m2, add_self_loop=False, add_3D=True):
    """
    :param m1: ligand molecule
    :param m2: pocket molecule
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return:
    complex: graphs contain m1, m2 and complex
    """
    # small molecule
    new_order1 = rdmolfiles.CanonicalRankAtoms(m1)
    mol1 = rdmolops.RenumberAtoms(m1, new_order1)

    # pocket
    new_order2 = rdmolfiles.CanonicalRankAtoms(m2)
    mol2 = rdmolops.RenumberAtoms(m2, new_order2)

    # construct graphs
    g1 = dgl.DGLGraph()  # small molecule
    g2 = dgl.DGLGraph()  # pocket

    # add nodes
    num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
    num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
    num_atoms = num_atoms_m1 + num_atoms_m2
    g1.add_nodes(num_atoms_m1)
    g2.add_nodes(num_atoms_m2)

    if add_self_loop:
        nodes1 = g1.nodes()
        g1.add_edges(nodes1, nodes1)
        nodes2 = g2.nodes()
        g2.add_edges(nodes2, nodes2)

    # add edges, ligand molecule
    num_bonds1 = mol1.GetNumBonds()
    src1 = []
    dst1 = []
    for i in range(num_bonds1):
        bond1 = mol1.GetBondWithIdx(i)
        u = bond1.GetBeginAtomIdx()
        v = bond1.GetEndAtomIdx()
        src1.append(u)
        dst1.append(v)
    src_ls1 = np.concatenate([src1, dst1])
    dst_ls1 = np.concatenate([dst1, src1])
    g1.add_edges(src_ls1, dst_ls1)

    # add edges, pocket
    num_bonds2 = mol2.GetNumBonds()
    src2 = []
    dst2 = []
    for i in range(num_bonds2):
        bond2 = mol2.GetBondWithIdx(i)
        u = bond2.GetBeginAtomIdx()
        v = bond2.GetEndAtomIdx()
        src2.append(u)
        dst2.append(v)
    src_ls2 = np.concatenate([src2, dst2])
    dst_ls2 = np.concatenate([dst2, src2])
    g2.add_edges(src_ls2, dst_ls2)


    # assign atom features
    # 'h', features of atoms
    g1.ndata['h'] = AtomFeaturizer(mol1)['h']
    g2.ndata['h'] = AtomFeaturizer(mol2)['h']

    # assign edge features
    # 'd', distance between ligand atoms
    dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
    g1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

    # 'd', distance between pocket atoms
    dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    g2_d = torch.tensor(dis_matrix_P[src_ls2, dst_ls2], dtype=torch.float).view(-1, 1)


    # efeats1
    efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
    g1.edata['e'] = torch.cat([efeats1[::2], efeats1[::2]])

    # efeats2
    efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
    g2.edata['e'] = torch.cat([efeats2[::2], efeats2[::2]])

    # 'e'
    g1.edata['e'] = torch.cat([g1.edata['e'], g1_d * 0.1], dim=-1)
    g2.edata['e'] = torch.cat([g2.edata['e'], g2_d * 0.1], dim=-1)

    if add_3D:
        g1.ndata['C'] = mol1.GetConformers()[0].GetPositions()
        g2.ndata['C'] = mol2.GetConformers()[0].GetPositions()

        # calculate the 3D info for g1
        src_nodes, dst_nodes = g1.find_edges(range(g1.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g1.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g1), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g1.edata['e'] = torch.cat([g1.edata['e'], D3_info_th], dim=-1)

        # calculate the 3D info for g2
        src_nodes, dst_nodes = g2.find_edges(range(g2.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g2.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g2), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g2.edata['e'] = torch.cat([g2.edata['e'], D3_info_th], dim=-1)

        g1.edata['e'] = torch.where(torch.isnan(g1.edata['e']), torch.full_like(g1.edata['e'], 0), g1.edata['e'])
        g2.edata['e'] = torch.where(torch.isnan(g2.edata['e']), torch.full_like(g2.edata['e'], 0), g2.edata['e'])


    return g1, g2

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))



def get_atom_features(atom, stereo, features, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    possible_atoms = ['C', 'N', 'O', 'S', 'F', 'P',
                      'Cl', 'Br', 'I', 'B', 'Si',
                      'Fe', 'Zn', 'Cu', 'Mn', 'Mo']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    atom_features += [int(i) for i in list("{0:06b}".format(features))]

    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:

        atom_features += [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(atom_features)


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """

    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)

def graphs_from_mol_v2(mol1, mol2, add_3D=True):
    """
    :param m1: ligand molecule
    :param m2: pocket molecule
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return:
    complex: graphs contain m1, m2
    """
    # # small molecule
    # new_order1 = rdmolfiles.CanonicalRankAtoms(m1)
    # mol1 = rdmolops.RenumberAtoms(m1, new_order1)
    #
    # # pocket
    # new_order2 = rdmolfiles.CanonicalRankAtoms(m2)
    # mol2 = rdmolops.RenumberAtoms(m2, new_order2)

    ##########################
    # small molecule         #
    ##########################
    g1 = dgl.DGLGraph()
    features1 = rdDesc.GetFeatureInvariants(mol1)

    stereo1 = Chem.FindMolChiralCenters(mol1)
    chiral_centers1 = [0] * mol1.GetNumAtoms()
    for i in stereo1:
        chiral_centers1[i[0]] = i[1]

    dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())

    g1.add_nodes(mol1.GetNumAtoms())
    node_features1 = []
    edge_features1 = []
    distace_features1 = []
    for i in range(mol1.GetNumAtoms()):

        atom_i = mol1.GetAtomWithIdx(i)
        atom_i_features = get_atom_features(atom_i, chiral_centers1[i], features1[i])
        node_features1.append(atom_i_features)

        for j in range(mol1.GetNumAtoms()):
            bond_ij = mol1.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                g1.add_edge(i, j)

                g1_dis_ij = torch.tensor(dis_matrix_L[i, j], dtype=torch.float)
                distace_features1.append(g1_dis_ij * 0.1)

                bond_features_ij = get_bond_features(bond_ij)
                edge_features1.append(bond_features_ij)

    g1.ndata['h'] = torch.from_numpy(np.array(node_features1))  # dgl添加原子/节点特征
    g1.edata['e'] = torch.from_numpy(np.array(edge_features1))  # dgl添加键/边特征
    g1.edata['e'] = torch.cat([g1.edata['e'], torch.from_numpy(np.array(distace_features1)).view(-1, 1)], dim=-1) #dgl添加距离信息


    ##########################
    # protein pocket         #
    ##########################
    g2 = dgl.DGLGraph()
    features2 = rdDesc.GetFeatureInvariants(mol2)

    stereo2 = Chem.FindMolChiralCenters(mol2)
    chiral_centers2 = [0] * mol2.GetNumAtoms()
    for i in stereo2:
        chiral_centers2[i[0]] = i[1]

    dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())

    g2.add_nodes(mol2.GetNumAtoms())
    node_features2 = []
    edge_features2 = []
    distace_features2 = []
    for i in range(mol2.GetNumAtoms()):

        atom_i = mol2.GetAtomWithIdx(i)
        atom_i_features = get_atom_features(atom_i, chiral_centers2[i], features2[i])
        node_features2.append(atom_i_features)

        for j in range(mol2.GetNumAtoms()):
            bond_ij = mol2.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                g2.add_edge(i, j)

                g2_dis_ij = torch.tensor(dis_matrix_P[i, j], dtype=torch.float)
                distace_features2.append(g2_dis_ij * 0.1)

                bond_features_ij = get_bond_features(bond_ij)
                edge_features2.append(bond_features_ij)


    g2.ndata['h'] = torch.from_numpy(np.array(node_features2))  # dgl添加原子/节点特征
    g2.edata['e'] = torch.from_numpy(np.array(edge_features2))  # dgl添加键/边特征
    g2.edata['e'] = torch.cat([g2.edata['e'], torch.from_numpy(np.array(distace_features2)).view(-1, 1)], dim=-1)  # dgl添加距离信息



    if add_3D:
        g1.ndata['C'] = torch.from_numpy(mol1.GetConformers()[0].GetPositions()).float()
        g2.ndata['C'] = torch.from_numpy(mol2.GetConformers()[0].GetPositions()).float()

        # calculate the 3D info for g1
        src_nodes, dst_nodes = g1.find_edges(range(g1.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g1.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g1), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g1.edata['e'] = torch.cat([g1.edata['e'], D3_info_th], dim=-1)

        # calculate the 3D info for g2
        src_nodes, dst_nodes = g2.find_edges(range(g2.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g2.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g2), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g2.edata['e'] = torch.cat([g2.edata['e'], D3_info_th], dim=-1)

        g1.edata['e'] = torch.where(torch.isnan(g1.edata['e']), torch.full_like(g1.edata['e'], 0), g1.edata['e'])
        g2.edata['e'] = torch.where(torch.isnan(g2.edata['e']), torch.full_like(g2.edata['e'], 0), g2.edata['e'])



    return g1, g2


#获得核心集的标签, 配体图，蛋白质图
############################2019###########################################
# def splitdata_core(PDB_code, LIG_G, PRO_G, labels):
#     '''
#     test set
#     '''
#     core_index = []
#     core_set = []
#     core_discompair = []
#     all = []
#     with open("2019coreset.txt", "r",encoding='gbk') as f:
#         for core_pro in f.readlines():
#             core_pro = core_pro.strip('\n').split()[0]
#             all.append(core_pro)
#             # if core_pro in PDB_code:     #找 coreset
#             if core_pro in PDB_code:
#                 tmp_index = PDB_code.index(core_pro)
#                 core_index.append(tmp_index)
#                 core_set.append(core_pro)
#             else:
#                 core_discompair.append(core_pro)
#
#     core_labels = [labels[i] for i in core_index]
#     lig_2019_coreset_graph = [LIG_G[i] for i in core_index]
#     pro_2019_coreset_graph = [PRO_G[i] for i in core_index]
#
#     return core_labels, lig_2019_coreset_graph, pro_2019_coreset_graph
#
#
# #def splitdata_train(PDB_code, LIG_G, PRO_G, labels, lig_coord, pro_coord):
# def splitdata_train(PDB_code, LIG_G, PRO_G, labels):
#     '''
#     train_set
#     '''
#     train_index = []
#     train_set = []
#     core_set = []
#     with open("2019coreset.txt", "r",encoding='gbk') as f:
#         for train_pro in f.readlines():
#             train_pro = train_pro.strip('\n').split()[0]
#             core_set.append(train_pro)
#         for train_line in PDB_code:
#             if train_line in core_set:
#                 pass
#             else:
#                 tmp_index = PDB_code.index(train_line)
#                 train_index.append(tmp_index)
#                 train_set.append(train_line)
#
#     train_labels = [labels[i] for i in train_index]
#     lig_2019_trainset_graph = [LIG_G[i] for i in train_index]
#     pro_2019_trainet_graph = [PRO_G[i] for i in train_index]
#     #lig_train_coord = [lig_coord[i] for i in train_index]
#     #pro_train_coord = [pro_coord[i] for i in train_index]
#     return train_labels,lig_2019_trainset_graph,pro_2019_trainet_graph

def splitdata_core(PDB_code, LIG_G, PRO_G, labels):
    '''
    test set
    '''
    core_index = []
    core_set = []
    core_discompair = []
    all = []
    with open("2016coreset.txt", "r",encoding='gbk') as f:
        for core_pro in f.readlines():
            core_pro = core_pro.strip('\n')
            all.append(core_pro)
            # if core_pro in PDB_code:     #找 coreset
            if core_pro in PDB_code:
                tmp_index = PDB_code.index(core_pro)
                core_index.append(tmp_index)
                core_set.append(core_pro)
            else:
                core_discompair.append(core_pro)

    core_labels = [labels[i] for i in core_index]
    lig_2016_coreset_graph = [LIG_G[i] for i in core_index]
    pro_2016_coreset_graph = [PRO_G[i] for i in core_index]

    return core_labels, lig_2016_coreset_graph, pro_2016_coreset_graph


#def splitdata_train(PDB_code, LIG_G, PRO_G, labels, lig_coord, pro_coord):
def splitdata_train(PDB_code, LIG_G, PRO_G, labels):
    '''
    train_set
    '''
    train_index = []
    train_set = []
    core_set = []
    with open("2016coreset.txt", "r",encoding='gbk') as f:
        for train_pro in f.readlines():
            train_pro = train_pro.strip('\n')
            core_set.append(train_pro)
        for train_line in PDB_code:
            if train_line in core_set:
                pass
            else:
                tmp_index = PDB_code.index(train_line)
                train_index.append(tmp_index)
                train_set.append(train_line)

    train_labels = [labels[i] for i in train_index]
    lig_2016_trainset_graph = [LIG_G[i] for i in train_index]
    pro_2016_trainet_graph = [PRO_G[i] for i in train_index]
    #lig_train_coord = [lig_coord[i] for i in train_index]
    #pro_train_coord = [pro_coord[i] for i in train_index]
    return train_labels,lig_2016_trainset_graph,pro_2016_trainet_graph




def pdbbind_v2016_main():
    root_dir_path = "F:/zouyr/pythonProject/DSBIN/data/pdbbind_v2016_refined/refined-set/"
    index_file = root_dir_path + 'index/INDEX_refined_set.2016'

    dataset = pdbbind(root_dir_path=root_dir_path, index_file=index_file)
    PDB_code, pKd, ligand_files, protein_files, ligands_loaded, protein_loaded = dataset.files2mol()

    log(f'Loading {len(PDB_code)} complexes to Graph.')
    lig_graphs = []
    pro_graphs = []
    for i in tqdm(range(len(PDB_code))):
        m1 = ligands_loaded[i][0]
        m2 = protein_loaded[i][0]
        g1, g2 = graphs_from_mol_v2(m1, m2, add_3D=True)
        lig_graphs.append(g1)
        pro_graphs.append(g2)

    train_labels, lig_2016_trainset_graph, pro_2016_trainet_graph = splitdata_train(PDB_code, lig_graphs, pro_graphs,
                                                                                    pKd)
    core_labels, lig_2016_coreset_graph, pro_2016_coreset_graph = splitdata_core(PDB_code, lig_graphs, pro_graphs,
                                                                                 pKd)

    # 保存pkd
    train_labels = np.array(train_labels)
    test_labels = np.array(core_labels)
    pKd = np.array(pKd)
    np.save('train_labels_2016', train_labels)
    np.save('test_labels_2016', test_labels)
    np.save('labels_2016', pKd)
    # 保存图
    save_graphs('lig_2016_graph', lig_graphs)
    save_graphs('pro_2016_graph', pro_graphs)
    save_graphs('lig_2016_trainset_graph', lig_2016_trainset_graph)
    save_graphs('pro_2016_trainset_graph', pro_2016_trainet_graph)
    save_graphs('lig_2016_testset_graph', lig_2016_coreset_graph)
    save_graphs('pro_2016_testset_graph', pro_2016_coreset_graph)



def get_pdbbind_v2016_general():
    root_dir_path = "F:/zouyr/pythonProject/DSBIN/data/pdbbind_v2016_general/"
    general_index_file = root_dir_path + 'index/INDEX_general_PL.2016'
    refined_index_file = root_dir_path + 'index/INDEX_refined_set.2016'

    def get_refined_code(index_file):
        with open(index_file, 'r', encoding='gbk') as f:
            PDB_code = []
            binding_data = []
            for line in f.readlines():
                if line[0] != "#":
                    splitted_elements = line.split()
                    PDB_code.append(splitted_elements[0])
                    splitted_pkd = splitted_elements[3]
                    data = splitted_pkd.split('=')[1]
                    binding_data.append(data)
            return PDB_code, binding_data

    def get_general_code(index_file):
        with open(index_file, 'r', encoding='gbk') as f:
            valid_pdb_code = []
            discompair_pdb_code = []
            valid_binding_data = []
            for line in f.readlines():
                if line[0] != "#":
                    splitted_elements = line.split()
                    splitted_pkd = splitted_elements[3]
                    if '=' in splitted_pkd:
                        valid_pdb_code.append(splitted_elements[0])
                        data = splitted_pkd.split('=')[1]
                        valid_binding_data.append(data)
                    else:
                        discompair_pdb_code.append(splitted_elements[0])
            return valid_pdb_code, valid_binding_data, discompair_pdb_code


    def getpKd(binding_data):
        labels = []
        for line in binding_data:
            unit = line[-2:]
            if unit == 'mM':
                pKd = - math.log10(float(line[:-2]) * 1e-03)
                # 保留两位小数
                pKd = round(pKd, 2)

            elif unit == 'uM':
                pKd = - math.log10(float(line[:-2]) * 1e-06)
                # 保留两位小数
                pKd = round(pKd, 2)

            elif unit == 'nM':
                pKd = - math.log10(float(line[:-2]) * 1e-09)
                pKd = round(pKd, 2)

            labels.append(pKd)
        return labels

    refined_pdb_code, refined_binding_data = get_refined_code(refined_index_file)
    general_valid_pdb_code, general_valid_binding_data, discompair_pdb_code = get_general_code(general_index_file)

    GenMinsRefined_code = []
    GenMinsRefined_bindingdata = []
    for i in range(len(general_valid_pdb_code)):
        if general_valid_pdb_code[i] not in refined_pdb_code:

            GenMinsRefined_code.append(general_valid_pdb_code[i])
            GenMinsRefined_bindingdata.append(general_valid_binding_data[i])

    ligand_files = [os.path.join(root_dir_path, "{}/".format(pdb), "{}_ligand.sdf".format(pdb)) for pdb in GenMinsRefined_code]
    protein_files = [os.path.join(root_dir_path, "{}/".format(pdb), "{}_pocket.pdb".format(pdb)) for pdb in GenMinsRefined_code]
    pKd = getpKd(GenMinsRefined_bindingdata)
    
    # del ligand_files[0]
    # del protein_files[0]
    # del pKd[0]


    return ligand_files, protein_files, pKd, (GenMinsRefined_code, pKd)



def get_general_mol(ligand_files,protein_files):
    log(f'Loading {len(ligand_files)} complexes of ligand.')
    ligands_loaded = []
    lig_to_remove = []
    for pdb in tqdm(ligand_files):
        try:
            lig = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if lig == None:
                lig_to_remove.append(pdb)
                continue
            ligands_loaded.append(lig)
        except:
            lig_to_remove.append(pdb)


    log(f'Loading {len(protein_files)} complexes of protein.')
    protein_loaded = []
    pro_to_remove = []
    for pdb in tqdm(protein_files):
        try:
            pro = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if pro == None:
                pro_to_remove.append(pdb)
                continue
            protein_loaded.append(pro)
        except:
            pro_to_remove.append(pdb)

def get_general_graph(ligand_files,protein_files):
    ligands_loaded, lig_to_remove, protein_loaded, pro_to_remove = get_general_mol(ligand_files,protein_files)
    lig_graphs = []
    pro_graphs = []
    invaild = []
    for i in tqdm(range(len(ligands_loaded))):
        m1 = ligands_loaded[i][0]
        m2 = protein_loaded[i][0]
        try:
            g1, g2 = graphs_from_mol_v2(m1, m2, add_3D=True)
            lig_graphs.append(g1)
            pro_graphs.append(g2)
        except:
            invaild.append(i)

    save_graphs('lig_general_2016_graph', lig_graphs)
    save_graphs('pro_general_2016_graph', pro_graphs)


def CSAR_NRC_HiQ_set():
    def file_name(file_dir):
        L = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                # 读取pdb文件
                if os.path.splitext(file)[1] == '.mol2':
                    L.append(os.path.join(root, file))
            return root, dirs, files, L

    # # 读取本地文件夹下pdb文件（../gg/）
    root, dirs, files, L = file_name('CSAR_NRC_HiQ_Set/Structures/set2')

    kd_data = []
    for dir in dirs:
        kd_index_file = 'CSAR_NRC_HiQ_Set/Structures/set2/' +  dir + '/kd.dat'
        with open(kd_index_file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                splitted_elements = line.strip().split(',')
                kd_data.append((splitted_elements[0],splitted_elements[1],splitted_elements[2]))

    ligand_files = [os.path.join('CSAR_NRC_HiQ_Set/Structures/set2/', "{}/".format(dir), "{}_ligand.pdb".format(dir)) for dir in
                    dirs]
    protein_files = [os.path.join('CSAR_NRC_HiQ_Set/Structures/set2/', "{}/".format(dir), "{}_protein.pdb".format(dir)) for dir in
                    dirs]


    log(f'Loading {len(ligand_files)} complexes of ligand.')
    ligands_loaded = []
    lig_to_remove = []
    for pdb in tqdm(ligand_files):
        try:
            lig = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if lig == None:
                lig_to_remove.append(pdb)
                continue
            ligands_loaded.append(lig)
        except:
            lig_to_remove.append(pdb)


    log(f'Loading {len(protein_files)} complexes of protein.')
    protein_loaded = []
    pro_to_remove = []
    for pdb in tqdm(protein_files):
        try:
            pro = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if pro == None:
                pro_to_remove.append(pdb)
                continue
            protein_loaded.append(pro)
        except:
            pro_to_remove.append(pdb)

    # #get graph
    # lig_graphs = []
    # pro_graphs = []
    # invaild = []
    # log(f'Loading {len(ligands_loaded)} complexes graph.')
    # for i in tqdm(range(len(ligands_loaded))):
    #     m1 = ligands_loaded[i][0]
    #     m2 = protein_loaded[i][0]
    #     try:
    #         g1, g2 = graphs_from_mol_v2(m1, m2, add_3D=True)
    #         lig_graphs.append(g1)
    #         pro_graphs.append(g2)
    #     except:
    #         invaild.append(i)


def get_pdbbind_v2020_holdout():
    index_file1 = 'INDEX_refined_set.2016'
    index_file2 = 'INDEX_refined_set.2020'

    def getcode(index_file):
        with open(index_file, 'r', encoding='gbk') as f:
            PDB_code = []
            binding_data = []
            for line in f.readlines():
                if line[0] != "#":
                    splitted_elements = line.split()
                    PDB_code.append(splitted_elements[0])
                    splitted_pkd = splitted_elements[3]
                    data = splitted_pkd.split('=')[1]
                    binding_data.append(data)
            return PDB_code, binding_data

    PDB_2016_code, binding_2016_data = getcode(index_file1)
    PDB_2020_code, binding_2020_data = getcode(index_file2)

    gen_code = []
    gen_bindingdata = []
    for i in range(len(PDB_2020_code)):
        if PDB_2020_code[i] not in PDB_2016_code:
            gen_code.append(PDB_2020_code[i])
            gen_bindingdata.append(binding_2020_data[i])

    df = pd.DataFrame(gen_code, columns=['PDB_code'])
    pdbs = df['PDB_code'].tolist()

    ligand_files = [os.path.join('refined-set/', "{}/".format(pdb), "{}_ligand.sdf".format(pdb)) for pdb in gen_code]
    protein_files = [os.path.join('refined-set/', "{}/".format(pdb), "{}_pocket.pdb".format(pdb)) for pdb in gen_code]

    # 取出pkd
    def getlabel(binding_data):
        labels = []
        for line in binding_data:
            unit = line[-2:]
            if unit == 'mM':
                pKd = - math.log10(float(line[:-2]) * 1e-03)
                # 保留两位小数
                pKd = round(pKd, 2)

            elif unit == 'uM':
                pKd = - math.log10(float(line[:-2]) * 1e-06)
                # 保留两位小数
                pKd = round(pKd, 2)

            elif unit == 'nM':
                pKd = - math.log10(float(line[:-2]) * 1e-09)
                pKd = round(pKd, 2)

            labels.append(pKd)
        return labels


    labels = getlabel(gen_bindingdata)
    np.save('labels_gen2020', labels)

    log(f'Loading {len(ligand_files)} complexes of ligand.')
    ligands_loaded = []
    lig_to_remove = []
    for pdb in tqdm(ligand_files):
        try:
            lig = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if lig == None:
                lig_to_remove.append(pdb)
                continue
            ligands_loaded.append(lig)
        except:
            lig_to_remove.append(pdb)

    log(f'Loading {len(protein_files)} complexes of protein.')
    protein_loaded = []
    pro_to_remove = []
    for pdb in tqdm(protein_files):
        try:
            pro = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if pro == None:
                pro_to_remove.append(pdb)
                continue
            protein_loaded.append(pro)
        except:
            pro_to_remove.append(pdb)
    lig_graphs = []
    pro_graphs = []
    invaild = []
    for i in tqdm(range(len(ligands_loaded))):
        m1 = ligands_loaded[i][0]
        m2 = protein_loaded[i][0]
        try:
            g1, g2 = graphs_from_mol_v2(m1, m2, add_3D=True)
            lig_graphs.append(g1)
            pro_graphs.append(g2)
        except:
            invaild.append(i)

    save_graphs('lig_gen2020_graph', lig_graphs)
    save_graphs('pro_gen2020_graph', pro_graphs)



    # root_dir_path = "F:/zouyr/pythonProject/DSBIN/data/pdbbind_v2016_general/"
    # general_index_file = root_dir_path + 'index/INDEX_general_PL.2016'
    # refined_index_file = 'F:/zouyr/pythonProject/DSBIN/data/pdbbind_v2020/index/INDEX_refined_set.2020'
    #
    # def get_refined_code(index_file):
    #     with open(index_file, 'r', encoding='gbk') as f:
    #         PDB_code = []
    #         binding_data = []
    #         for line in f.readlines():
    #             if line[0] != "#":
    #                 splitted_elements = line.split()
    #                 PDB_code.append(splitted_elements[0])
    #                 splitted_pkd = splitted_elements[3]
    #                 data = splitted_pkd.split('=')[1]
    #                 binding_data.append(data)
    #         return PDB_code, binding_data
    #
    # def get_general_code(index_file):
    #     with open(index_file, 'r', encoding='gbk') as f:
    #         valid_pdb_code = []
    #         discompair_pdb_code = []
    #         valid_binding_data = []
    #         for line in f.readlines():
    #             if line[0] != "#":
    #                 splitted_elements = line.split()
    #                 splitted_pkd = splitted_elements[3]
    #                 if '=' in splitted_pkd:
    #                     valid_pdb_code.append(splitted_elements[0])
    #                     data = splitted_pkd.split('=')[1]
    #                     valid_binding_data.append(data)
    #                 else:
    #                     discompair_pdb_code.append(splitted_elements[0])
    #         return valid_pdb_code, valid_binding_data, discompair_pdb_code
    #
    #
    # def getpKd(binding_data):
    #     labels = []
    #     for line in binding_data:
    #         unit = line[-2:]
    #         if unit == 'mM':
    #             pKd = - math.log10(float(line[:-2]) * 1e-03)
    #             # 保留两位小数
    #             pKd = round(pKd, 2)
    #
    #         elif unit == 'uM':
    #             pKd = - math.log10(float(line[:-2]) * 1e-06)
    #             # 保留两位小数
    #             pKd = round(pKd, 2)
    #
    #         elif unit == 'nM':
    #             pKd = - math.log10(float(line[:-2]) * 1e-09)
    #             pKd = round(pKd, 2)
    #
    #         labels.append(pKd)
    #     return labels
    #
    # refined_pdb_code, refined_binding_data = get_refined_code(refined_index_file)
    # general_valid_pdb_code, general_valid_binding_data, discompair_pdb_code = get_general_code(general_index_file)
    #
    # pdb_code = [x for x in refined_pdb_code  if x not in general_valid_pdb_code]

def get_newdata_graph(file_path):
    file_path = 'F:/zouyr/pythonProject/MGNN-main/data/raw/SARS-CoV-BA'
    def file_name(file_dir):
        L = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                # 读取pdb文件
                if os.path.splitext(file)[1] == '.sdf':
                    L.append(os.path.join(root, file))
            return root, dirs, files, L

    # # 读取本地文件夹下pdb文件（../gg/）
    root, dirs, files, L = file_name(file_path)



    protein_files = []
    ligand_files = []
    for dir in dirs:
        file_dir = os.path.join(file_path + "/{}/".format(dir))
        for r, d, f in os.walk(file_dir):
            for file in f:
                if file.split('_')[-1] == 'pocket.pdb':
                    pocket_file = r  + file
                    protein_files.append(pocket_file)
                elif file.split('_')[-1] == 'ligand.pdb':
                    ligand_file = r  + file
                    ligand_files.append(ligand_file)
                elif file.split('_')[-1] == 'ligand.sdf':
                    ligand_file = r  + file
                    ligand_files.append(ligand_file)



    log(f'Loading {len(ligand_files)} complexes of ligand.')
    ligands_loaded = []
    lig_to_remove = []
    for pdb in tqdm(ligand_files):
        try:
            lig = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if lig == None:
                lig_to_remove.append(pdb)
                continue
            ligands_loaded.append(lig)
        except:
            lig_to_remove.append(pdb)


    log(f'Loading {len(protein_files)} complexes of protein.')
    protein_loaded = []
    pro_to_remove = []
    for pdb in tqdm(protein_files):
        try:
            pro = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if pro == None:
                pro_to_remove.append(pdb)
                continue
            protein_loaded.append(pro)
        except:
            pro_to_remove.append(pdb)

    #get graph
    lig_graphs = []
    pro_graphs = []
    invaild = []
    log(f'Loading {len(ligands_loaded)} complexes graph.')
    for i in tqdm(range(len(ligands_loaded))):
        m1 = ligands_loaded[i][0]
        m2 = protein_loaded[i][0]
        try:
            g1, g2 = graphs_from_mol_v2(m1, m2, add_3D=True)
            lig_graphs.append(g1)
            pro_graphs.append(g2)
        except:
            invaild.append(i)

    df = pd.read_csv('pkd.csv',header=None)
    df = np.array(df)
    pkd = []
    for dir in dirs:
        for i in range(len(df)):
            if df[i][0] == dir:
                pkd.append(df[i][1])

    pKd = np.array(pkd)
    np.save('labels_sars', pKd)
    # 保存图
    save_graphs('lig_sars_graph', lig_graphs)
    save_graphs('pro_sars_graph', pro_graphs)


def construt_graph():


    PDB = pd.read_csv('pkd.csv',header=None)
    PDB_ID = np.array(PDB[0])
    pkd = np.array(PDB[1])



    file_path = 'F:/zouyr/pythonProject/MGNN-main/data/raw/SARS-CoV-BA'

    protein_files = []
    ligand_files = []
    for dir in PDB_ID:
        file_dir = os.path.join(file_path + "/{}/".format(dir))
        for r, d, f in os.walk(file_dir):
            for file in f:
                if file.split('_')[-1] == 'pocket.pdb':
                    pocket_file = r  + file
                    protein_files.append(pocket_file)
                elif file.split('_')[-1] == 'ligand.pdb':
                    ligand_file = r  + file
                    ligand_files.append(ligand_file)
                elif file.split('_')[-1] == 'ligand.sdf':
                    ligand_file = r  + file
                    ligand_files.append(ligand_file)


    log(f'Loading {len(ligand_files)} complexes of ligand.')
    ligands_loaded = []
    lig_to_remove = []
    for pdb in tqdm(ligand_files):
        try:
            lig = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if lig == None:
                lig_to_remove.append(pdb)
                continue
            ligands_loaded.append(lig)
        except:
            lig_to_remove.append(pdb)


    log(f'Loading {len(protein_files)} complexes of protein.')
    protein_loaded = []
    pro_to_remove = []
    for pdb in tqdm(protein_files):
        try:
            pro = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if pro == None:
                pro_to_remove.append(pdb)
                continue
            protein_loaded.append(pro)
        except:
            pro_to_remove.append(pdb)

    #get graph
    lig_graphs = []
    pro_graphs = []
    invaild = []
    log(f'Loading {len(ligands_loaded)} complexes graph.')
    for i in tqdm(range(len(ligands_loaded))):
        m1 = ligands_loaded[i][0]
        m2 = protein_loaded[i][0]
        try:
            g1, g2 = graphs_from_mol_v2(m1, m2, add_3D=True)
            lig_graphs.append(g1)
            pro_graphs.append(g2)
        except:
            invaild.append(i)



    pKd = np.array(pkd)
    np.save('labels_sars', pKd)
    # 保存图
    save_graphs('lig_sars_graph', lig_graphs)
    save_graphs('pro_sars_graph', pro_graphs)

def get_casf2016():

    root_dir_path = "F:/zouyr/pythonProject/CASF-2016/coreset/"
    index_file = "F:/zouyr/pythonProject/CASF-2016/power_docking/CoreSet.dat"

    PDB_code = []
    PKD = []
    with open(index_file, 'r') as f:
        for line in f.readlines():
            if line[0] != "#":
                splitted_elements = line.split()
                PDB_code.append(splitted_elements[0])
                splitted_pkd = float(splitted_elements[3])
                PKD.append(splitted_pkd)
    df = pd.DataFrame(PDB_code, columns=['PDB_code'])
    PDB_code = df['PDB_code'].tolist()

    ligand_files = [os.path.join(root_dir_path, "{}/".format(pdb), "{}_ligand.sdf".format(pdb)) for pdb in PDB_code]
    protein_files = [os.path.join(root_dir_path, "{}/".format(pdb), "{}_pocket.pdb".format(pdb)) for pdb in PDB_code]


    log(f'Loading {len(ligand_files)} complexes of ligand.')
    ligands_loaded = []
    lig_to_remove = []
    for pdb in tqdm(ligand_files):
        try:
            lig = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if lig == None:
                lig_to_remove.append(pdb)
                continue
            ligands_loaded.append(lig)
        except:
            lig_to_remove.append(pdb)


    log(f'Loading {len(protein_files)} complexes of protein.')
    protein_loaded = []
    pro_to_remove = []
    for pdb in tqdm(protein_files):
        try:
            pro = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if pro == None:
                pro_to_remove.append(pdb)
                continue
            protein_loaded.append(pro)
        except:
            pro_to_remove.append(pdb)

    #get graph
    lig_graphs = []
    pro_graphs = []
    invaild = []
    log(f'Loading {len(ligands_loaded)} complexes graph.')
    for i in tqdm(range(len(ligands_loaded))):
        m1 = ligands_loaded[i][0]
        m2 = protein_loaded[i][0]
        try:
            g1, g2 = graphs_from_mol_v2(m1, m2, add_3D=True)
            lig_graphs.append(g1)
            pro_graphs.append(g2)
        except:
            invaild.append(i)


    pKd = np.array(PKD)
    np.save('labels_casf2016', pKd)
    # 保存图
    save_graphs('lig_casf2016_graph', lig_graphs)
    save_graphs('pro_casf2016_graph', pro_graphs)

def get_casf2013():

    root_dir_path = "F:/zouyr/pythonProject/CASF-2016/CASF-2013-updated/coreset/"
    index_file = "F:/zouyr/pythonProject/CASF-2016/CASF-2013-updated/coreset/index/2013_core_data.lst"

    PDB_code = []
    PKD = []
    with open(index_file, 'r') as f:
        for line in f.readlines():
            if line[0] != "#":
                splitted_elements = line.split()
                PDB_code.append(splitted_elements[0])
                splitted_pkd = float(splitted_elements[3])
                PKD.append(splitted_pkd)
    df = pd.DataFrame(PDB_code, columns=['PDB_code'])
    PDB_code = df['PDB_code'].tolist()

    ligand_files = [os.path.join(root_dir_path, "{}/".format(pdb), "{}_ligand.sdf".format(pdb)) for pdb in PDB_code]
    protein_files = [os.path.join(root_dir_path, "{}/".format(pdb), "{}_protein.pdb".format(pdb)) for pdb in PDB_code]


    log(f'Loading {len(ligand_files)} complexes of ligand.')
    ligands_loaded = []
    lig_to_remove = []
    for pdb in tqdm(ligand_files):
        try:
            lig = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if lig == None:
                lig_to_remove.append(pdb)
                continue
            ligands_loaded.append(lig)
        except:
            lig_to_remove.append(pdb)


    log(f'Loading {len(protein_files)} complexes of protein.')
    protein_loaded = []
    pro_to_remove = []
    for pdb in tqdm(protein_files):
        try:
            pro = load_molecule(pdb,
                                sanitize=False,
                                remove_hs=True)
            if pro == None:
                pro_to_remove.append(pdb)
                continue
            protein_loaded.append(pro)
        except:
            pro_to_remove.append(pdb)

    #get graph
    lig_graphs = []
    pro_graphs = []
    invaild = []
    log(f'Loading {len(ligands_loaded)} complexes graph.')
    for i in tqdm(range(len(ligands_loaded))):
        m1 = ligands_loaded[i][0]
        m2 = protein_loaded[i][0]
        try:
            g1, g2 = graphs_from_mol_v2(m1, m2, add_3D=True)
            lig_graphs.append(g1)
            pro_graphs.append(g2)
        except:
            invaild.append(i)


    pKd = np.array(PKD)
    np.save('labels_casf2013', pKd)
    # 保存图
    save_graphs('lig_casf2013_graph', lig_graphs)
    save_graphs('pro_casf2013_graph', pro_graphs)










if __name__ == '__main__':
    print('a')

# R2 0.9172725263201752  MAE Val_loss 1.6449213 rmse 0.5153537819570039 sd 0.514988126137987 Rp 0.9578081142090177