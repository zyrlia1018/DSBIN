from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import r2_score
import math
from math import sqrt

loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

class Nomalizer(object):
    def __init__(self, ndarray):
        self.mean = ndarray.mean()
        self.std = ndarray.std()

    def norm(self, ndarray):
        return (ndarray - self.mean) / self.std

    def denorm(self, normed_ndarray):
        return normed_ndarray * self.std + self.mean


def get_loss(model, data_loader):
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []
    for ligand_graphs, protein_graphs, ligand_lens, protein_lens, labels in tqdm(data_loader):
        outputs, interaction_map,weightblock = model([ligand_graphs.to(device),
                                         protein_graphs.to(device),
                                         torch.tensor(ligand_lens).to(device),
                                         torch.tensor(protein_lens).to(device)])
        
        loss = loss_fn(outputs, torch.tensor(labels).to(device).float())
        mae_loss = mae_loss_fn(outputs, torch.tensor(labels).to(device).float())
        valid_outputs += outputs.cpu().detach().numpy().tolist()
        valid_loss.append(loss.cpu().detach().numpy())
        valid_mae_loss.append(mae_loss.cpu().detach().numpy())
        valid_labels += labels
    
        
    loss = np.mean(np.array(valid_loss).flatten())
    mae_loss = np.mean(np.array(valid_mae_loss).flatten())   
    rmse1 = rmse(np.array(valid_labels).reshape(-1,), np.array(valid_outputs).reshape(-1,))
    r = pearson(np.array(valid_labels).reshape(-1,), np.array(valid_outputs).reshape(-1,))
   # R2 = r2_score(np.array(valid_labels),np.array(valid_outputs))
   # R = math.sqrt(R2)
    return loss, mae_loss, rmse1, r

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    best_mae_loss = 100
    #best_val_r = 0
    for epoch in range(max_epochs):
        model.train()
        running_loss = []
        weightblock_list = []
        tq_loader = tqdm(train_loader)
        for samples in tq_loader:
            optimizer.zero_grad()
            outputs, interaction_map, weightblock = model([samples[0].to(device),
                                              samples[1].to(device),
                                              torch.tensor(samples[2]).to(device),
                                              torch.tensor(samples[3]).to(device)])
            # L2 regularization was also  added on the interaction map
            l1_norm = torch.norm(interaction_map, p=2) * 1e-4
            loss = mae_loss_fn(outputs, torch.tensor(samples[4]).to(device).float()) + l1_norm
            loss.backward()
            optimizer.step()
            loss = loss - l1_norm
            running_loss.append(loss.cpu().detach())
            weightblock = weightblock.cpu().detach().numpy()
            weightblock_list.append(weightblock)
            tq_loader.set_description(
                "Epoch: " + str(epoch + 1) + "  Training loss: " + str(np.mean(np.array(running_loss))))
        model.eval()
        val_loss, mae_loss, rmse1, r = get_loss(model, valid_loader)
        scheduler.step(mae_loss)
        W = weightblock
        print(" Epoch: " + str(epoch + 1) +
              " Rp " + str(r) +
              " train_loss " + str(np.mean(np.array(running_loss))) +
              " Val_loss " + str(val_loss) +
              " MAE Val_loss " + str(mae_loss)+
              " w " + str(W))
       # if r > best_val_r:
       #     best_val_r = r
       #     torch.save(model.state_dict(),"best_model_r.tar")
        if mae_loss < best_mae_loss:
            best_mae_loss = mae_loss
            torch.save(model.state_dict(),"best_model_noh2omae.tar")
        with open('result_NOH2O_mae.txt', 'a') as f:
            f.write(str(epoch + 1)+" W: "+ str(W)  +" RUN_LOSS "+str(np.mean(np.array(running_loss)))+" VAL_LOSS "+str(val_loss)+" MAE_LOSS "+str(mae_loss)+ " Rp "+ str(r) + "\n")
