


import numpy as np

from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set,NNConv,GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F

from Graphbymol import PLDistance

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

'''
We did not use it in this task due to the high demand for computing resources
'''

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
            self.register_parameter('gamma', None)
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
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x


def get_non_lin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    elif type == 'relu':
        return nn.ReLU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


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


def apply_norm(g, h, norm_type, norm_layer):
    if norm_type == 'GN':
        return norm_layer(g, h)
    return norm_layer(h)


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


def cross_attention(queries, keys, values, mask, cross_msgs):
    """Compute cross attention.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    Args:
      queries: NxD float tensor --> queries
      keys: MxD float tensor --> keys
      values: Mxd
      mask: NxM
    Returns:
      attention_x: Nxd float tensor.
    """
    if not cross_msgs:
        return queries * 0.
    a = mask * torch.mm(queries, torch.transpose(keys, 1, 0)) - 1000. * (1. - mask)
    a_x = torch.softmax(a, dim=1)  # i->j, NxM, a_x.sum(dim=1) = torch.ones(N)
    attention_x = torch.mm(a_x, values)  # (N,d)
    return attention_x


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
                 leakyrelu_neg_slope,
                 nonlin,
                 dropout,
                 layer_norm,
                 final_h_layer_norm,
                 node_input_dim = 59,
                 edge_input_dim = 14,
                 node_hidden_dim = 59,
                 edge_hidden_dim = 59,
                 num_step_message_passing = 6
                 ):
        super(GatherModel, self).__init__()
        self.node_input_dim = node_input_dim
        self.final_h_layer_norm = final_h_layer_norm
        self.num_step_message_passing = num_step_message_passing

        self.linear0 = nn.Linear(node_input_dim,node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim,2,1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(nn.Linear(edge_input_dim,edge_hidden_dim),nn.ReLU(),
                                     nn.Linear(edge_hidden_dim,node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_feats=node_hidden_dim,
                           out_feats=node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type='sum',
                           residual=True)

        self.att_mlp_Q_lig = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_input_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_lig = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_input_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_lig = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_input_dim, bias=False),
        )
        self.att_mlp_Q_rec = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_input_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K_rec = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_input_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V_rec = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_input_dim, bias=False),
        )


        self.final_h_layernorm_layer_lig = get_norm(self.final_h_layer_norm, node_hidden_dim)
        self.final_h_layernorm_layer_rec = get_norm(self.final_h_layer_norm, node_hidden_dim)

        self.node_mlp_lig = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden_dim),
            get_layer_norm(layer_norm, node_hidden_dim),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(node_input_dim, node_input_dim),
            get_layer_norm(layer_norm, node_hidden_dim),
        )

        self.node_mlp_rec = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden_dim),
            get_layer_norm(layer_norm, node_hidden_dim),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            nn.Dropout(dropout),
            nn.Linear(node_input_dim, node_input_dim),
            get_layer_norm(layer_norm, node_hidden_dim),
        )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)


    def forward(self, lig_graph, lig_n_feat, lig_e_feat, rec_graph, rec_n_feat, rec_e_feat, mask):
        with lig_graph.local_scope() and rec_graph.local_scope():

            #ligand_mpnn
            lig_init = lig_n_feat.clone()
            lig_out = F.relu(self.linear0(lig_n_feat))
            for i in range(self.num_step_message_passing):
                if lig_e_feat is not None:
                    lig_m = torch.relu(self.conv(lig_graph, lig_out, lig_e_feat))
                else:
                    lig_m = torch.relu(self.conv.bias + self.conv.res_fc(lig_out))
                lig_out = self.message_layer(torch.cat([lig_m, lig_out], dim=1))
                # Fv represents the final atomic feature for each atom v
            lig_Fv = lig_out + lig_init

            #receptor_mpnn
            rec_init = rec_n_feat.clone()
            rec_out = F.relu(self.linear0(rec_n_feat))
            for i in range(self.num_step_message_passing):
                if rec_e_feat is not None:
                    rec_m = torch.relu(self.conv(rec_graph, rec_out, rec_e_feat))
                else:
                    rec_m = torch.relu(self.conv.bias + self.conv.res_fc(rec_out))
                rec_out = self.message_layer(torch.cat([rec_m, rec_out], dim=1))
                # Fv represents the final atomic feature for each atom v
            rec_Fv = rec_out + rec_init

            h_feats_lig_norm = apply_norm(lig_graph, lig_Fv, self.final_h_layer_norm, self.final_h_layernorm_layer_lig)
            h_feats_rec_norm = apply_norm(rec_graph, rec_Fv, self.final_h_layer_norm, self.final_h_layernorm_layer_rec)
            cross_attention_lig_feat = cross_attention(self.att_mlp_Q_lig(h_feats_lig_norm),
                                                       self.att_mlp_K(h_feats_rec_norm),
                                                       self.att_mlp_V(h_feats_rec_norm), mask, self.cross_msgs)
            cross_attention_rec_feat = cross_attention(self.att_mlp_Q(h_feats_rec_norm),
                                                       self.att_mlp_K_lig(h_feats_lig_norm),
                                                       self.att_mlp_V_lig(h_feats_lig_norm), mask.transpose(0, 1),
                                                       self.cross_msgs)
            cross_attention_lig_feat = apply_norm(lig_graph, cross_attention_lig_feat, self.final_h_layer_norm,
                                                  self.final_h_layernorm_layer_lig)
            cross_attention_rec_feat = apply_norm(rec_graph, cross_attention_rec_feat, self.final_h_layer_norm,
                                                  self.final_h_layernorm_layer_rec)

            return cross_attention_lig_feat, cross_attention_rec_feat





class DSBINModel(nn.Module):
    """
    This the main class for DSBIN model
    """
    def __init__(self,
                 node_input_dim=59,
                 edge_input_dim=14,
                 node_hidden_dim=59,
                 edge_hidden_dim=59,
                 num_step_message_passing = 6,
                 interaction = 'dot',
                 num_step_set2set = 2,
                 num_layer_set2set = 1):
        super(DSBINModel, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.gather = GatherModel(
                 node_input_dim = self.node_input_dim,
                 edge_input_dim = self.edge_input_dim,
                 node_hidden_dim =self.node_hidden_dim,
                 edge_hidden_dim =self.edge_hidden_dim,
                 num_step_message_passing = 6,
                 leakyrelu_neg_slope= 0.01,
                 nonlin = 'lkyrelu',
                 dropout= 0.1,
                 layer_norm= 'LN',
                 final_h_layer_norm = 'GN',
        )
        # self.receptor_gather = GatherModel(
        #          node_input_dim = self.node_input_dim,
        #          edge_input_dim = self.edge_input_dim,
        #          node_hidden_dim =self.node_hidden_dim,
        #          edge_hidden_dim =self.edge_hidden_dim,
        #          num_step_message_passing = 6,
        #          leakyrelu_neg_slope= 0.01,
        #          nonlin = 'lkyrelu',
        #          dropout= 0.1,
        #          layer_norm= 'LN',
        #          final_h_layer_norm = 'GN',
        # )

        ####################################################################################
        self.fc_weight1 = nn.Linear(10,10)
        self.fc_weight2 = nn.Linear(10,10)
        self.fc1 = nn.Linear(10 * 4 * self.node_hidden_dim,256)
        #self.bn1 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(64, 1)
        self.imap= nn.Linear(80, 1)

        self.num_step_set2set = num_step_set2set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_ligand = Set2Set(node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_protein = Set2Set(node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)


    def forward(self, data):
        ligand = data[0]
        protein = data[1]
        ligand_len = data[2]
        protein_len = data[3]
        lig_n_feat = ligand.ndata['x'].float()
        lig_e_feat = ligand.edata['w'].float()
        rec_n_feat = protein.ndata['x'].float()
        rec_e_feat = protein.edata['w'].float()
        coords_lig = ligand.ndata['C']
        coords_rec = protein.ndata['C']
        dis_matrix = PLDistance(coords_lig, coords_rec)

        # mask = None
        mask = get_mask(ligand.batch_num_nodes(), protein.batch_num_nodes(), self.device)

        cross_attention_lig_feat, cross_attention_rec_feat = self.gather(lig_graph=ligand,
                                                                         lig_n_feat=lig_n_feat,
                                                                         lig_e_feat=lig_e_feat,
                                                                         rec_graph=protein,
                                                                         rec_n_feat=rec_n_feat,
                                                                         rec_e_feat=rec_e_feat,
                                                                         mask=mask
                                                                         )


        # node embeddings after interaction phase
        ligand_features = cross_attention_lig_feat.float()
        protein_features = cross_attention_rec_feat.float()


        # Interaction phase
        #torch.mm(a, b)是矩阵a和b矩阵相乘
        len_map = torch.mm(ligand_len.t(), protein_len)
        #print(weight.size())
        #print(len_map.size())
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

        interaction_map = torch.mul(interaction_map, dis_matrix.to(device))
        protein_prime = torch.mm(interaction_map.t(), ligand_features)
        ligand_prime = torch.mm(interaction_map, protein_features)
        ligand_features = ligand_prime
        protein_features = protein_prime

        ligand_features = self.set2set_ligand(ligand, ligand_features)
        protein_features = self.set2set_protein(protein, protein_features)

        final_features = torch.cat((ligand_features, protein_features), 1)



        predictions = torch.relu(self.fc1(final_features))
        #predictions = self.dp1(predictions)
        predictions = torch.relu(self.fc2(predictions))
       # predictions = self.dp2(predictions)
        predictions = self.fc3(predictions)


        return predictions, ret_interaction_map















