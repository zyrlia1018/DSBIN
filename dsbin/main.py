# python imports
import pandas as pd
import warnings
import os
import argparse

# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem

# torch imports
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# dgl imports
import dgl
from dgl.data.utils import load_graphs

# local imports
from model_ones import DSBINModel
from train import train
from utils import *

import warnings 
warnings.filterwarnings("ignore")

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='DSBIN', help="The name of the current project: default:DSBIN")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | "
                                          "tanh-general", default='dot')
parser.add_argument('--max_epochs', required=False, default=500, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, default=1, help="The batch size for training")

args = parser.parse_args()
project_name = args.name
interaction = args.interaction
max_epochs = int(args.max_epochs)
batch_size = int(args.batch_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu")
#if not os.path.isdir("runs/run-" + str(project_name)):
#    os.makedirs("./runs/run" + str(project_name))
#    os.makedirs("./runs/run" + str(project_name) + "/model")

def collate(samples):
    ligand_graphs, protein_graphs, labels = map(list, zip(*samples))
    ligand_graphs = dgl.batch(ligand_graphs)
    protein_graphs = dgl.batch(protein_graphs)
    ligand_len_matrix = get_len_matrix(ligand_graphs.batch_num_nodes())
    protein_len_matrix = get_len_matrix(protein_graphs.batch_num_nodes())
    return ligand_graphs, protein_graphs, ligand_len_matrix, protein_len_matrix, labels

class Dataclass(Dataset):
    def __init__(self, ligand_graphs, protein_graphs, labels):
        self.ligand_graphs = ligand_graphs
        self.protein_graphs = protein_graphs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):

        ligand_graph = self.ligand_graphs[item]
        protein_graph = self.protein_graphs[item]
        delta_g = self.labels[item]

        return [ligand_graph, protein_graph, delta_g]


class Nomalizer(object):
    def __init__(self, ndarray):
        self.mean = ndarray.mean()
        self.std = ndarray.std()

    def norm(self, ndarray):
        return (ndarray - self.mean) / self.std

    def denorm(self, normed_ndarray):
        return normed_ndarray * self.std + self.mean


def main():
    train_ligand_graphs = load_graphs('lig_2016_trainset_graph')[0]
    train_protein_graphs = load_graphs('pro_2016_trainset_graph')[0]
    train_pkd = np.load('train_labels_2016.npy')

    test_ligand_graphs = load_graphs('lig_2016_testset_graph')[0]
    test_protein_graphs = load_graphs('pro_2016_testset_graph')[0]
    test_pkd = np.load('test_labels_2016.npy')
    
    #Norm = Nomalizer(train_pkd)
    #train_pkd = Norm.norm(train_pkd) 
    #test_pkd = Norm.norm(test_pkd)


    train_dataset = Dataclass(ligand_graphs=train_ligand_graphs,protein_graphs=train_protein_graphs,labels=train_pkd)
    #分出来
    valid_dataset = Dataclass(ligand_graphs=test_ligand_graphs,protein_graphs=test_protein_graphs,labels=test_pkd)

    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=batch_size)

    model = DSBINModel(interaction=interaction)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)

    #trainTEST(max_epochs, model, optimizer, train_loader)
    train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)


if __name__ == '__main__':
    main()





