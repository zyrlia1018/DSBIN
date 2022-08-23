import numpy as np
from copy import deepcopy
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set,NNConv,GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

def PLDistance(lig_coord,pro_coord):
    distance_matrix = cdist(lig_coord, pro_coord, 'euclidean')
    return distance_matrix  #np.array()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def norm(x, eps=1e-8):
    mean = x.mean(dim=0, keepdim=True)
    var = x.std(dim=0, keepdim=True)
    return (x - mean) / (var + eps)

#图标准化
class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)    #可以使用给定的名称作为属性访问参数
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h, node_type):
        graph_size = g.batch_num_nodes(node_type) if self.is_node else g.batch_num_edges(node_type)
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0 )

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x

def get_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_norm_type == 'GN':
        return GraphNorm(dim)
    else:
        assert layer_norm_type == '0' or layer_norm_type == 0
        return nn.Identity()


class CoordsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coords):
        norm = coords.norm(dim=-1, keepdim=True)
        normed_coords = coords / norm.clamp(min=self.eps)
        return normed_coords * self.scale

def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes, device):
    rows = ligand_batch_num_nodes.sum()
    cols = receptor_batch_num_nodes.sum()
    mask = torch.zeros(rows, cols, device=device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask

class GatherModel(nn.Module):
    """
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 40.
    edge_input_dim : int
        Dimension of input edge feature, default to be 10.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 40.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    """
    def __init__(self,
                 node_input_dim,
                 edge_input_dim,
                 node_hidden_dim,
                 edge_hidden_dim,
                 num_step_message_passing
                 ):
        super(GatherModel, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing

        self.linear0 = nn.Linear(self.node_input_dim, self.node_hidden_dim)
        self.set2set = Set2Set(self.node_hidden_dim,2,1)
        self.message_layer = nn.Linear(2 * self.node_hidden_dim, self.node_hidden_dim)
        edge_network = nn.Sequential(nn.Linear(self.edge_input_dim,self.edge_hidden_dim),nn.ReLU(),
                                     nn.Linear(self.edge_hidden_dim,self.node_hidden_dim * self.node_hidden_dim))
        self.conv = NNConv(in_feats=self.node_hidden_dim,
                           out_feats=self.node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type='sum',
                           residual=True)

    def forward(self, g, n_feat, e_feat):
        """Returns the node embeddings after message passing phase.
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        Returns
        -------
        res : node features
        """
        init = n_feat.clone()
        out = F.relu(self.linear0(n_feat))
        for i in range(self.num_step_message_passing):
            if e_feat is not None:
                m = torch.relu(self.conv(g,out,e_feat))
            else:
                m = torch.relu(self.conv.bias + self.conv.res_fc(out))
            out = self.message_layer(torch.cat([m, out], dim=1))
            #Fv represents the final atomic feature for each atom v
        Fv = out + init
        return Fv




class DSBINModel(nn.Module):
    """
    This the main class for DSBIN model
    """
    def __init__(self,
                 device,
                 node_input_dim,
                 edge_input_dim,
                 node_hidden_dim,
                 edge_hidden_dim,
                 num_step_message_passing,
                 interaction = 'dot',
                 num_step_set2set = 2,
                 num_layer_set2set = 1,
                 cutoff=10
                 ):
        super(DSBINModel, self).__init__()
        self.device = device
        self.cutoff = cutoff
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.ligand_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                         self.node_hidden_dim, self.edge_input_dim,
                                         self.num_step_message_passing,
                                         )
        self.protein_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                          self.node_hidden_dim, self.edge_input_dim,
                                          self.num_step_message_passing,
                                          )

        self.fc_weight1 = nn.Linear(10,10)
        self.fc_weight2 = nn.Linear(10,10)
        self.linear_layer = nn.Linear(10, 10)
        self.fc1 = nn.Linear(10 * 4 * self.node_hidden_dim,256)
        self.dp1 = nn.Dropout(0.1)
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256,64)
        self.bn2 = nn.LayerNorm(64)
        self.dp2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 1)


        self.num_step_set2set = num_step_set2set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_ligand = Set2Set(self.node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_protein = Set2Set(self.node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

    def shell_matrix(self,ligand,protein):

        ligand_Coord = ligand.ndata['C']
        protein_Coord = protein.ndata['C']

        mask = get_mask(ligand.batch_num_nodes(), protein.batch_num_nodes(), self.device)
        distance_matrix = torch.from_numpy(PLDistance(ligand_Coord,protein_Coord)).to(self.device)
        real_distance = mask * distance_matrix
        zero_matrix = torch.zeros(np.shape(real_distance))

        shell_neig = []
        # 1A范围内的邻接矩阵
        for i in range(self.cutoff):
            matrix = deepcopy(zero_matrix)
            shelli = np.where((real_distance > i) & (real_distance <= i+1))
            matrix[shelli]=1
            shell_neig.append(matrix)

        return shell_neig

    def forward(self, data):
        ligand = data[0]
        protein = data[1]
        ligand_len = data[2]
        protein_len = data[3]

        shell_neig = self.shell_matrix(ligand,protein)
        # weight = torch.tensor([x.sum() for x in shell_neig]).squeeze(-1)
        # weight_norm = norm(weight)
        # # linear_layer = nn.Linear(len(weight_norm),len(weight_norm))
        # weightblock = torch.sigmoid(self.linear_layer(weight_norm))
        w = torch.ones(10).to(device)
        w = torch.relu(self.fc_weight1(w))
        weightblock = torch.sigmoid(self.fc_weight2(w))


        # node embeddings after interaction phase
        ligand_features = self.ligand_gather(ligand, ligand.ndata['x'].float(), ligand.edata['w'].float())
        try:
            # if edge exists in a molecule
            protein_features = self.protein_gather(protein, protein.ndata['x'].float(), protein.edata['w'].float())
        except:
            # if edge doesn't exist in a molecule, for example in case of water
            protein_features = self.protein_gather(protein, protein.ndata['x'].float(), None)

        # Interaction phase
        len_map = torch.mm(ligand_len.t(), protein_len)

        if 'dot' not in self.interaction:
            X1 = ligand_features.unsqueeze(0)
            Y1 = protein_features.unsqueeze(1)
            X2 = X1.repeat(protein_features.shape[0], 1, 1)
            Y2 = Y1.repeat(1, ligand_features.shape[0], 1)
            Z = torch.cat([X2, Y2], -1)

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2)
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t())
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map = torch.mm(ligand_features, protein_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.node_hidden_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map)
            interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map)

        interaction_all = []
        final_features = []
        for i in range(len(weightblock)):
            interaction_map_i = torch.mul(interaction_map, shell_neig[i].to(device)) * (weightblock[i].to(device))
            protein_features_tmp = torch.mm(interaction_map_i.t(), ligand_features)
            ligand_features_tmp = torch.mm(interaction_map_i, protein_features)
            ligand_features_i = self.set2set_ligand(ligand, ligand_features_tmp)
            protein_features_i = self.set2set_protein(protein, protein_features_tmp)
            final_features_i = torch.cat((ligand_features_i, protein_features_i), 1)
            interaction_all.append(interaction_map_i)
            final_features.append(final_features_i)


        # Prediction phase
        final_features =  torch.cat([x for x in final_features], 1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = self.bn1(predictions)
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.bn2(predictions)
        predictions = self.fc3(predictions)

        return predictions, ret_interaction_map, weightblock















