import torch
from tqdm import tqdm
import torch
from sklearn.metrics import r2_score
import math
from math import sqrt

from dgl.data.utils import load_graphs
import numpy as np
from model_ones import DSBINModel
from main import Dataclass,DataLoader,collate
from sklearn.linear_model import LinearRegression

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp


state_dict = torch.load('./best_model_mae.tar', map_location='cpu')


model=DSBINModel()
model.load_state_dict(state_dict)
model.to(device)


test_ligand_graphs = load_graphs('lig_2016_trainset_graph')[0]
test_protein_graphs = load_graphs('pro_2016_trainset_graph')[0]
test_pkd = np.load('train_labels_2016.npy')



valid_dataset = Dataclass(ligand_graphs=test_ligand_graphs,protein_graphs=test_protein_graphs,labels=test_pkd)
test_valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=1)

valid_outputs = []
valid_labels = []
valid_loss = []
valid_mae_loss = []
interaction_maps = []
for ligand_graphs, protein_graphs, ligand_lens, protein_lens, labels in tqdm(
        test_valid_loader):
    outputs, interaction_map,weightblock = model([ligand_graphs.to(device),
                                      protein_graphs.to(device),
                                      torch.tensor(ligand_lens).to(device),
                                      torch.tensor(protein_lens).to(device)])
    loss = loss_fn(outputs.cpu(), torch.tensor(labels).float())
    mae_loss = mae_loss_fn(outputs.cpu(), torch.tensor(labels).float())
    valid_outputs += outputs.cpu().detach().numpy().round(2).tolist()
    valid_loss.append(loss.cpu().detach().numpy())
    valid_mae_loss.append(mae_loss.cpu().detach().numpy())
    valid_labels += labels
    weightblock.cpu().detach().numpy()
    # interaction_maps.append(interaction_map)

loss = np.mean(np.array(valid_loss).flatten())
mae_loss = np.mean(np.array(valid_mae_loss).flatten())
R2 = r2_score(np.array(valid_labels), np.array(valid_outputs))
rmse = rmse(np.array(valid_labels).reshape(-1,), np.array(valid_outputs).reshape(-1,))
sd = sd(np.array(valid_labels).reshape(-1,), np.array(valid_outputs).reshape(-1,))
# intermap = interaction_maps.cpu().detach().numpy()
tmp = np.array(valid_outputs)
tmp2 = np.array(valid_labels)
np.save('train0.742_pre_gen.npy',tmp)
np.save('train0.742_true_gen.npy',tmp2)


r = pearson(np.array(valid_labels).reshape(-1,), np.array(valid_outputs).reshape(-1,))

print(" R2 "+str(R2)," MAE Val_loss " + str(mae_loss)+" rmse "+ str(rmse)+" sd "+str(sd)+ " r "+str(r)+" w  "+str(weightblock) )
