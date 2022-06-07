import os
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import cdist
from dgl.data.utils import save_graphs

from Graphbymol import getGraph,getDAGraph,load_molecule,getPLDIAGraph,getCoordGraph
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def files():
    root_dir_path = "../data/pdbbind_v2019_refined/refined-set/"
    index_file = root_dir_path + 'index/INDEX_refined_set.2019'
    PDB_code = []
    binding_data = []

    with open(index_file, 'r',encoding='gbk') as f:
        for line in f.readlines():
            if line[0] != "#":
                splitted_elements = line.split()
                PDB_code.append(splitted_elements[0])
                splitted_pkd = splitted_elements[3]
                data = splitted_pkd.split('=')[1]
                binding_data.append(data)
    df = pd.DataFrame(PDB_code, columns=['PDB_code'])
    pdbs = df['PDB_code'].tolist()
    # binding_data = df['binding_data'].tolist()

    ligand_files = [os.path.join(root_dir_path, "{}/".format(pdb), "{}_ligand.sdf".format(pdb)) for pdb in PDB_code]
    protein_files = [os.path.join(root_dir_path, "{}/".format(pdb), "{}_pocket.pdb".format(pdb)) for pdb in PDB_code]
    return PDB_code,binding_data,ligand_files,protein_files


#取出pkd
def getlabel(binding_data):
    labels = []
    for line in binding_data:
        unit = line[-2:]
        if unit == 'mM':
            pKd = - math.log10(float(line[:-2]) * 1e-03)
            pKd = round(pKd, 2)

        elif unit == 'uM':
            pKd = - math.log10(float(line[:-2]) * 1e-06)
            pKd = round(pKd, 2)

        elif unit == 'nM':
            pKd = - math.log10(float(line[:-2]) * 1e-09)
            pKd = round(pKd, 2)

        labels.append(pKd)
    return labels

#protein与ligand的距离矩阵性质图
def pld_LIG_G(ligand_files,protein_files):
    LIG_DA_G = []
    for i in range(len(ligand_files)):
        lig_mol = load_molecule(ligand_files[i])[0]
        pro_mol = load_molecule(protein_files[i])[0]
        G = getPLDIAGraph(lig_mol,pro_mol)
        LIG_DA_G.append(G)
    return LIG_DA_G


#ligand内部的距离矩阵性质图
def DisAtom_LIG_G(ligand_files):
    LIG_DA_G = []
    lig_coord = []
    for ligand in ligand_files:
        file_path = ligand
        mol_coord = load_molecule(file_path)
        mol = mol_coord[0]
        coord = mol_coord[1]
        G = getDAGraph(mol)
        LIG_DA_G.append(G)
        lig_coord.append(coord)
    return LIG_DA_G,lig_coord

def Atom_LIG_G(ligand_files):
    LIG_G = []
    lig_coord = []
    for ligand in ligand_files:
        file_path = ligand
        mol_coord = load_molecule(file_path)
        mol = mol_coord[0]
        coord = mol_coord[1]
        G = getGraph(mol)
        LIG_G.append(G)
        lig_coord.append(coord)
    return LIG_G,lig_coord

def Atom_PRO_G(protein_files):
    PRO_G = []
    pro_coord = []
    for protein in protein_files:
        file_path = protein
        mol_coord = load_molecule(file_path)
        mol = mol_coord[0]
        coord = mol_coord[1]
        G_p = getGraph(mol)
        PRO_G.append(G_p)
        pro_coord.append(coord)
    return PRO_G, pro_coord

def AtomCoord_LIG_G(ligand_files):
    LIG_G = []
    for ligand in ligand_files:
        file_path = ligand
        mol_coord = load_molecule(file_path,remove_hs=True)
        mol = mol_coord[0]
        G_p = getCoordGraph(mol)
        LIG_G.append(G_p)
    return LIG_G

def AtomCoord_PRO_G(protein_files):
    PRO_G = []
    for protein in protein_files:
        file_path = protein
        mol_coord = load_molecule(file_path,remove_hs=True)
        mol = mol_coord[0]
        G_p = getCoordGraph(mol)
        PRO_G.append(G_p)
    return PRO_G

#def splitdata_core(PDB_code, LIG_G, PRO_G, labels, lig_coord, pro_coord):
def splitdata_core(PDB_code, LIG_G, PRO_G, labels):
    '''
    test set
    '''
    core_index = []
    core_set = []
    core_discompair = []
    all = []
    with open("../data/pdbbind_v2019_refined/2019coreset.txt", "r",encoding='gbk') as f:
        for core_pro in f.readlines():
            core_pro = core_pro.strip('\n')
            all.append(core_pro)
            # if core_pro in PDB_code:
            if core_pro in PDB_code:
                tmp_index = PDB_code.index(core_pro)
                core_index.append(tmp_index)
                core_set.append(core_pro)
            else:
                core_discompair.append(core_pro)

    core_labels = [labels[i] for i in core_index]
    lig_2019_coreset_graph = [LIG_G[i] for i in core_index]
    pro_2019_coreset_graph = [PRO_G[i] for i in core_index]
    #lig_core_coord = [lig_coord[i] for i in core_index]
    #pro_core_coord = [pro_coord[i] for i in core_index]
    #return core_labels,lig_2016_coreset_graph,pro_2016_coreset_graph,lig_core_coord,pro_core_coord
    return core_labels, lig_2019_coreset_graph, pro_2019_coreset_graph


#def splitdata_train(PDB_code, LIG_G, PRO_G, labels, lig_coord, pro_coord):
def splitdata_train(PDB_code, LIG_G, PRO_G, labels):
    '''
    train_set
    '''
    train_index = []
    train_set = []
    core_set = []
    with open("../data/pdbbind_v2019_refined/2019coreset.txt", "r",encoding='gbk') as f:
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
    lig_2019_trainset_graph = [LIG_G[i] for i in train_index]
    pro_2019_trainet_graph = [PRO_G[i] for i in train_index]
    #lig_train_coord = [lig_coord[i] for i in train_index]
    #pro_train_coord = [pro_coord[i] for i in train_index]
    return train_labels,lig_2019_trainset_graph,pro_2019_trainet_graph

def Distance(lig_coord,pro_coord):
    distance_matrix = cdist(lig_coord, pro_coord, 'euclidean')
    return distance_matrix  #np.array()

def main():
    PDB_code, binding_data, ligand_files, protein_files = files()
    labels = getlabel(binding_data)
    LIG_G = AtomCoord_LIG_G(ligand_files)
    PRO_G = AtomCoord_PRO_G(protein_files)
    train_labels,lig_2016_trainset_graph,pro_2016_trainset_graph = \
        splitdata_train(PDB_code,LIG_G,PRO_G,labels)

    core_labels,lig_2016_coreset_graph,pro_2016_coreset_graph = \
        splitdata_core(PDB_code,LIG_G,PRO_G,labels)

    train_labels = np.array(train_labels)
    test_labels = np.array(core_labels)
    np.save('train_labels_2019',train_labels)
    np.save('test_labels_2019',test_labels)

    save_graphs('lig_2019_trainset_graph', lig_2016_trainset_graph)
    save_graphs('pro_2019_trainset_graph', pro_2016_trainset_graph)
    save_graphs('lig_2019_testset_graph', lig_2016_coreset_graph)
    save_graphs('pro_2019_testset_graph', pro_2016_coreset_graph)

    #lig_train_coord = np.array(lig_train_coord)
    #pro_train_coord = np.array(pro_train_coord)
    #lig_test_coord = np.array(lig_core_coord)
    #pro_test_coord = np.array(pro_core_coord)
    #np.save('lig_train_coord', lig_train_coord)
    #np.save('pro_train_coord', pro_train_coord)
    #np.save('lig_test_coord', lig_test_coord)
    #np.save('pro_test_coord', pro_test_coord)




if __name__ == '__main__':
    main()
