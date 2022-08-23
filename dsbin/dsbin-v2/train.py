from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from sklearn.metrics import r2_score
import math
from math import sqrt
from scipy import stats

loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



# def rmse(y_true, y_pred):
#     dev = np.square(y_true.ravel() - y_pred.ravel())
#     return np.sqrt(np.sum(dev) / y_true.shape[0])
#
#
def pcc(y_true, y_pred):
    p = stats.pearsonr(y_true, y_pred)
    return p[0]

#####



def sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

# def pearson(y,f):
#     rp = np.corrcoef(y, f)[0,1]
#     return rp


def rmse(y_pred, y_true):
    rmse = sqrt(((y_pred - y_true)**2).mean(axis=0))
    return rmse

def pearson(vector1, vector2):
    n = len(vector1)

    # simple sums

    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))

    # sum up the squares

    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])

    # sum up the products

    p_sum = sum([vector1[i] * vector2[i] for i in range(n)])

    # 分子num，分母den

    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))

    if den == 0:
        return 0.0

    return num / den

class Nomalizer(object):
    def __init__(self, ndarray):
        self.mean = ndarray.mean()
        self.std = ndarray.std()

    def norm(self, ndarray):
        return (ndarray - self.mean) / self.std

    def denorm(self, normed_ndarray):
        return normed_ndarray * self.std + self.mean

def z_loss_func(epoch, y_pred, y_true, gate=0.6):
    z_loss = ( 1-gate) * rmse(y_pred, y_true) + gate * (1 - pearson(y_pred, y_true))
    if epoch < 300:
        loss = z_loss
    else:
        loss = (1 - pearson(y_pred, y_true))
    return loss





def get_loss(model, data_loader):
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []
    for ligand_graphs, protein_graphs, ligand_lens, protein_lens, labels in tqdm(data_loader):   #tqdm(list)方法可以传入
        outputs, interaction_map,weightblock = model([ligand_graphs.to(device),
                                         protein_graphs.to(device),
                                         torch.tensor(ligand_lens).to(device),
                                         torch.tensor(protein_lens).to(device)])
        
        loss = loss_fn(outputs, torch.tensor(labels).to(device).float())
        mae_loss = mae_loss_fn(outputs, torch.tensor(labels).to(device).float())
        valid_outputs += outputs.cpu().detach().numpy().tolist()  #detach的方法，将variable参数从网络中隔离开，不参与参数更新。
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

def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader,project_name):
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
            loss = z_loss_func(epoch=epoch,y_pred=outputs.reshape(-1,), y_true=torch.tensor(samples[4]).reshape(-1,),gate=0.6) + l1_norm
            loss.backward()
            optimizer.step()
            loss = loss - l1_norm
            running_loss.append(loss)
            weightblock_list.append(weightblock.detach().numpy())
            tq_loader.set_description(
                "Epoch: " + str(epoch + 1) + "  Training loss: " + str(loss.detach().numpy()))
        model.eval()
        val_loss, mae_loss, rmse, r = get_loss(model, valid_loader)
        scheduler.step(r)
        W = np.mean(np.array(weightblock_list),axis=0)
        print(" Epoch: " + str(epoch + 1) +
              " Rp " + str(r) +
              " RMSD " + str(rmse) +
              " Val_loss " + str(val_loss) +
              " MAE Val_loss " + str(mae_loss)+
              " w " + str(W))
       # if r > best_val_r:
       #     best_val_r = r
       #     torch.save(model.state_dict(),"best_model_r.tar")
        if mae_loss < best_mae_loss:
            best_mae_loss = mae_loss
            torch.save(model.state_dict(),"best_model.tar")
        with open('result.txt', 'a') as f:
            f.write(str(epoch + 1)+" W: "+ str(W) +" VAL_LOSS "+str(val_loss)+" MAE_LOSS "+str(mae_loss)+ " Rp "+ str(r) + "\n")
