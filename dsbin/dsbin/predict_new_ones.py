
import torch
from torch.utils.data import DataLoader, Dataset
import dgl
import argparse
from tqdm import tqdm
import torch
import numpy as np

import math
from math import sqrt
from model_ones import DSBINModel
from main import Dataclass,DataLoader,collate
from sklearn.linear_model import LinearRegression
from utils import *

from Graphbymol import getGraph,getDAGraph,load_molecule,getPLDIAGraph,getCoordGraph
from Graphbymol import PLDistance
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', required=False, default=1, help="The batch size for training")
args = parser.parse_args()
batch_size = int(args.batch_size)

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device('cpu')


def AtomCoord_LIG_G(ligand_files):
    LIG_G = []
    for ligand in ligand_files:
        file_path = ligand
        mol_coord = load_molecule(file_path,remove_hs=True)
        mol = mol_coord[0]
        G_p = getCoordGraph(mol)
        LIG_G.append(G_p)
    return LIG_G

#包含属性和坐标的蛋白质图
def AtomCoord_PRO_G(protein_files):
    PRO_G = []
    for protein in protein_files:
        file_path = protein
        mol_coord = load_molecule(file_path,remove_hs=True)
        mol = mol_coord[0]
        G_p = getCoordGraph(mol)
        PRO_G.append(G_p)
    return PRO_G

def collate(samples):
    ligand_graphs, protein_graphs = map(list, zip(*samples))
    ligand_graphs = dgl.batch(ligand_graphs)
    protein_graphs = dgl.batch(protein_graphs)
    ligand_len_matrix = get_len_matrix(ligand_graphs.batch_num_nodes())
    protein_len_matrix = get_len_matrix(protein_graphs.batch_num_nodes())
    return ligand_graphs, protein_graphs, ligand_len_matrix, protein_len_matrix


class Dataclass(Dataset):
    def __init__(self, ligand_graphs, protein_graphs):
        self.ligand_graphs = ligand_graphs
        self.protein_graphs = protein_graphs


    def __len__(self):
        return len(self.protein_graphs)

    def __getitem__(self, item):
        ligand_graph = self.ligand_graphs[item]
        protein_graph = self.protein_graphs[item]

        return [ligand_graph, protein_graph]

def predict(model, data_loader):
    valid_outputs = []
    interaction_maps = []
    for ligand_graphs, protein_graphs, ligand_lens, protein_lens in tqdm(data_loader):
        outputs, interaction_map,weightblock = model([ligand_graphs.to(device),
                                         protein_graphs.to(device),
                                         torch.tensor(ligand_lens).to(device),
                                         torch.tensor(protein_lens).to(device)])

        valid_outputs += outputs.cpu().detach().numpy().tolist()
        interaction_map.cpu().detach().numpy()
        weightblock.cpu().detach().numpy()
       # interaction_maps.append(interaction_map)
   # intermap = np.array(interaction_maps)
    return interaction_map, valid_outputs, weightblock



# input data
ligand_files = ['./predict_data/ligand.pdb']
ligand_files = ligand_files
protein_files = ['./predict_data/protein.pdb']
protein_files = protein_files


LIG_G = AtomCoord_LIG_G(ligand_files)
PRO_G = AtomCoord_PRO_G(protein_files)


valid_dataset = Dataclass(ligand_graphs=LIG_G,protein_graphs=PRO_G)
valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=batch_size)


loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()


state_dict = torch.load('best_model_mae_0.741.tar', map_location=device)
#state_dict = torch.load('best_5637model_mae.tar', map_location=device)


model=DSBINModel()
model.load_state_dict(state_dict)
model.to(device)



intermap, valid_ourputs, weightblock = predict(model,valid_loader)
print(valid_ourputs)
#np.save('interaction_map',intermap)
#print(intermap)


#save intermap(demo)
matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8, matrix9, matrix10 = \
                    model.weight_matrix(LIG_G[0],PRO_G[0])
print(matrix1.sum())
print(matrix2.sum())
print(matrix3.sum())
print(matrix4.sum())
print(matrix5.sum())
print(matrix6.sum())
print(matrix7.sum())
print(matrix8.sum())
print(matrix9.sum())
print(matrix10.sum())


m1 = matrix1*weightblock[0]
m2 = matrix2*weightblock[1]
m3 = matrix3*weightblock[2]
m4 = matrix4*weightblock[3]
m5 = matrix5*weightblock[4]
m6 = matrix6*weightblock[5]
m7 = matrix7*weightblock[6]
m8 = matrix8*weightblock[7]
m9 = matrix9*weightblock[8]
m10 = matrix10*weightblock[9]
print(m10.sum())

dm = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8 + m9 + m10
imap = dm*intermap
fd = imap.cpu().detach().numpy()

np.save("intermap",fd)
