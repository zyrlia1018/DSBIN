# 导入pymol
from Graphbymol import getGraph,getDAGraph,load_molecule,getPLDIAGraph,getCoordGraph
import pymol
from pymol import *
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.mol2':
                L.append(os.path.join(root , file))
        return root, dirs, files, L
root, dirs, files, L =file_name('F:/zouyr/CSAR_HiQ_NRC_set/CSAR_NRC_HiQ_Set/Structures/set1')

for lines in dirs:
    pdb_file = os.path.join(root,lines)
    pdb_file = pdb_file.replace('\\', '/')
    _,_,_,L = file_name(pdb_file)
    L = L[0].replace('\\', '/')
    cmd.load(L)
    cmd.select("resn INH")
    cmd.save(pdb_file+"/{}_ligand.pdb".format(lines), "resn INH")
    cmd.remove("resn INH")
    cmd.remove("hydro")
    cmd.h_add('all')
    cmd.remove('h. and (e. c extend 1)')
    cmd.save(pdb_file+"/{}_protein.pdb".format(lines))
    cmd.delete('all')
    cmd.load(pdb_file+"/{}_ligand.pdb".format(lines))
    cmd.h_add('all')
    cmd.save(pdb_file + "/{}_ligand.pdb".format(lines))
    cmd.delete('all')


LIG_G = []
PRO_G = []
labels = []
for lines in dirs:
    pdb_file = os.path.join(root,lines)
    pdb_file = pdb_file.replace('\\', '/')

    #get lig_graph
    lig_mol_coord = load_molecule(pdb_file+"/{}_ligand.pdb".format(lines))
    lig_mol = lig_mol_coord[0]
    tmp1 = getCoordGraph(lig_mol)
    LIG_G.append(tmp1)

    #get pro_graph
    pro_mol_coord = load_molecule(pdb_file+"/{}_protein.pdb".format(lines))
    pro_mol = pro_mol_coord[0]
    tmp1 = getCoordGraph(pro_mol)
    PRO_G.append(tmp1)

    #get_labels
    data = pd.read_csv(pdb_file + "/kd.dat")
    tmp3 = float(data.columns[2])
    labels.append(tmp3)




from dgl.data.utils import save_graphs
import numpy as np
labels = np.array(labels)
np.save('../ZUIZHONGBAN/CSAR_NRC_HiQ_Set/Set1/set1_labels', labels)
save_graphs('../ZUIZHONGBAN/CSAR_NRC_HiQ_Set/Set1/lig_set1_graph',LIG_G)
save_graphs('../ZUIZHONGBAN/CSAR_NRC_HiQ_Set/Set1/pro_set1_graph',PRO_G)


