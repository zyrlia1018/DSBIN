from feature import *
from dgl import DGLGraph
import numpy as np
from scipy.spatial.distance import cdist
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    pass

def get_atom_features(atom):
    node_fea_tmp = atom_type_one_hot(atom)  #10
    node_fea_tmp += atom_degree_one_hot(atom)  #7
    node_fea_tmp += atom_explicit_valence_one_hot(atom)  #6
    node_fea_tmp += atom_implicit_valence_one_hot(atom)  #7
    node_fea_tmp += atom_hybridization_one_hot(atom)  #5
    node_fea_tmp += atom_formal_charge_one_hot(atom)  #5
    node_fea_tmp += atom_num_radical_electrons_one_hot(atom) #5
    node_fea_tmp += atom_is_aromatic_one_hot(atom) #2
    node_fea_tmp += atom_is_in_ring_one_hot(atom)  #2
    node_fea_tmp += atom_chirality_type_one_hot(atom) #3
    node_fea_tmp = list(map(int, node_fea_tmp))
    #node_fea_tmp += atom_mass(atom_i)
    return node_fea_tmp

def get_bond_features(bond):
    edge_fea_tmp = bond_type_one_hot(bond)
    edge_fea_tmp += bond_is_conjugated_one_hot(bond)
    edge_fea_tmp += bond_is_in_ring_one_hot(bond)
    edge_fea_tmp += bond_stereo_one_hot(bond)
    edge_fea_tmp += bond_direction_one_hot(bond)

def Distance(coordinates):
    distance_matrix = cdist(coordinates, coordinates, 'euclidean')
    return torch.from_numpy(distance_matrix)  #np.array()

def DistanceMat(lig_coordinates):
    distance_mat = []
    for line in lig_coordinates:
        distance_matrix = cdist(line, line, 'euclidean')
        distance_mat.append(distance_matrix)
    return np.array(distance_mat)

def feature_normalize(data):
    return (data - data.min())/(data.max()-data.min())


def Distance2center(lig_coordinates,pro_coordinates):
    pro2center = []
    for lig_xyz in lig_coordinates:
        ori_lig_center = np.mean(lig_xyz, axis=0)
        for pro_xyz in pro_coordinates:
            distance = torch.tensor(np.sqrt(((ori_lig_center - pro_xyz)**2).sum(axis=1)))
            weight = feature_normalize(1/distance)
            weight = weight.view(-1, 1, 1)
        pro2center.append(weight)
    return pro2center

def load_molecule(molecule_file, sanitize=False, calc_charges=False,
                  remove_hs=False, use_conformation=True):
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None, None

    if use_conformation:
        coordinates = get_mol_3d_coordinates(mol)
    else:
        coordinates = None

    return mol, coordinates

def get_mol_3d_coordinates(mol):
    """Get 3D coordinates of the molecule.

    This function requires that molecular conformation has been initialized.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.

    Returns
    -------
    numpy.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. For failures in getting the conformations, None will be returned.
    """

    try:
        conf = mol.GetConformer()
        conf_num_atoms = conf.GetNumAtoms()
        mol_num_atoms = mol.GetNumAtoms()
        assert mol_num_atoms == conf_num_atoms, \
            'Expect the number of atoms in the molecule and its conformation ' \
            'to be the same, got {:d} and {:d}'.format(mol_num_atoms, conf_num_atoms)
        return conf.GetPositions()
    except:
        warnings.warn('Unable to get conformation of the molecule.')
        return None

def get_norm_dismat(distance_matrix,node_input_dim=51):
    nodes = distance_matrix.shape[0]
    linear0 = nn.Linear(nodes,node_input_dim)
    out = F.relu(linear0(distance_matrix))
    return out


def getDAGraph(mol):
    G = DGLGraph()
    G.add_nodes(mol.GetNumAtoms())

    coords = get_mol_3d_coordinates(mol)
    distance_matrix = Distance(coords)
    distance_matrix = get_norm_dismat(distance_matrix)

    node_features = []
    edge_features = []
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        node_fea_tmp = get_atom_features(atom_i)
        #
        # 汇总
        node_features.append((node_fea_tmp))

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edge(i, j)
                # print(i,j)
                edge_fea_tmp = bond_type_one_hot(bond_ij)
                edge_fea_tmp += bond_is_conjugated_one_hot(bond_ij)
                edge_fea_tmp += bond_is_in_ring_one_hot(bond_ij)
                edge_fea_tmp += bond_stereo_one_hot(bond_ij)
                edge_fea_tmp += bond_direction_one_hot(bond_ij)
                edge_fea_tmp = list(map(int, edge_fea_tmp))
                edge_features.append(edge_fea_tmp)

    node_features = torch.from_numpy(np.array(node_features))
    edge_features = torch.from_numpy(np.array(edge_features))
    G.ndata['x'] = node_features  # dgl添加原子/节点特征
    G.ndata['D'] = distance_matrix # dgl添加距离embedding矩阵
    G.edata['w'] = edge_features  # dgl添加键/边特征
    return G

def getGraph(mol):
    G = DGLGraph()
    G.add_nodes(mol.GetNumAtoms())
    node_features = []
    edge_features = []
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        node_fea_tmp = get_atom_features(atom_i)
        #
        # 汇总
        node_features.append((node_fea_tmp))

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edge(i, j)
                # print(i,j)
                edge_fea_tmp = bond_type_one_hot(bond_ij)
                edge_fea_tmp += bond_is_conjugated_one_hot(bond_ij)
                edge_fea_tmp += bond_is_in_ring_one_hot(bond_ij)
                edge_fea_tmp += bond_stereo_one_hot(bond_ij)
                edge_fea_tmp += bond_direction_one_hot(bond_ij)
                edge_fea_tmp = list(map(int, edge_fea_tmp))
                edge_features.append(edge_fea_tmp)

    node_features = torch.from_numpy(np.array(node_features))
    edge_features = torch.from_numpy(np.array(edge_features))
    G.ndata['x'] = node_features  # dgl添加原子/节点特征
    G.edata['w'] = edge_features  # dgl添加键/边特征
    return G


def PLDistance(lig_coord,pro_coord):
    distance_matrix = cdist(lig_coord, pro_coord, 'euclidean')
    return distance_matrix  #np.array()



def getPLDIAGraph(lig_mol,pro_mol):
    G = DGLGraph()
    G.add_nodes(lig_mol.GetNumAtoms())

    lig_coord = get_mol_3d_coordinates(lig_mol)
    pro_coord = get_mol_3d_coordinates(pro_mol)
    PL_distance = []
    for i in range(len(lig_coord)):
        tmp = PLDistance(lig_coord[i], pro_coord[i])
        tmp[(tmp > 0) & (tmp < 3)] = 1.0
        tmp[(tmp > 3) & (tmp < 5)] = 0.8
        tmp[(tmp > 5) & (tmp < 10)] = 0.6
        tmp[(tmp > 8) & (tmp < 10)] = 0.4
        tmp[tmp > 10] = 0
        PL_distance.append(tmp)


    node_features = []
    edge_features = []
    for i in range(lig_mol.GetNumAtoms()):
        atom_i = lig_mol.GetAtomWithIdx(i)
        node_fea_tmp = get_atom_features(atom_i)
        #
        # 汇总
        node_features.append((node_fea_tmp))

        for j in range(lig_mol.GetNumAtoms()):
            bond_ij = lig_mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edge(i, j)
                # print(i,j)
                edge_fea_tmp = bond_type_one_hot(bond_ij)
                edge_fea_tmp += bond_is_conjugated_one_hot(bond_ij)
                edge_fea_tmp += bond_is_in_ring_one_hot(bond_ij)
                edge_fea_tmp += bond_stereo_one_hot(bond_ij)
                edge_fea_tmp += bond_direction_one_hot(bond_ij)
                edge_fea_tmp = list(map(int, edge_fea_tmp))
                edge_features.append(edge_fea_tmp)

    node_features = torch.from_numpy(np.array(node_features))
    edge_features = torch.from_numpy(np.array(edge_features))
    PL_distance = torch.from_numpy(np.array(PL_distance))
    G.ndata['x'] = node_features  # dgl添加原子/节点特征
    G.ndata['D'] = PL_distance # dgl添加距离embedding矩阵
    G.edata['w'] = edge_features  # dgl添加键/边特征
    return G

def getCoordGraph(mol):
    G = DGLGraph()
    G.add_nodes(mol.GetNumAtoms())

    coords = get_mol_3d_coordinates(mol)

    node_features = []
    edge_features = []
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        node_fea_tmp = get_atom_features(atom_i)
        #
        # 汇总
        node_features.append((node_fea_tmp))

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edge(i, j)
                # print(i,j)
                edge_fea_tmp = bond_type_one_hot(bond_ij)
                edge_fea_tmp += bond_is_conjugated_one_hot(bond_ij)
                edge_fea_tmp += bond_is_in_ring_one_hot(bond_ij)
                edge_fea_tmp += bond_stereo_one_hot(bond_ij)
                edge_fea_tmp += bond_direction_one_hot(bond_ij)
                edge_fea_tmp = list(map(int, edge_fea_tmp))
                edge_features.append(edge_fea_tmp)

    node_features = torch.from_numpy(np.array(node_features))
    edge_features = torch.from_numpy(np.array(edge_features))
    coords = torch.from_numpy(np.array(coords))
    G.ndata['x'] = node_features  # dgl添加原子/节点特征
    G.ndata['C'] = coords # dgl添加coordinates坐标
    G.edata['w'] = edge_features  # dgl添加键/边特征
    return G
