import numpy as np

from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set,NNConv,GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F

from Graphbymol import PLDistance

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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
                 node_input_dim = 52,
                 edge_input_dim = 14,
                 node_hidden_dim = 52,
                 edge_hidden_dim = 52,
                 num_step_message_passing = 4,
                 ):
        super(GatherModel, self).__init__()
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
                 node_input_dim=52,
                 edge_input_dim=14,
                 node_hidden_dim=52,
                 edge_hidden_dim=52,
                 num_step_message_passing = 4,
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


    def weight_matrix(self,ligand,protein):

        ligand_Coord = ligand.ndata['C'].cpu().float()
        protein_Coord = protein.ndata['C'].cpu().float()


        #distance_matrix = PLDistance(ligand_Coord,protein_Coord)

        matrix1 = PLDistance(ligand_Coord,protein_Coord)
        matrix1[(matrix1 > 0) & (matrix1 <= 1)] = 1
        matrix1[(matrix1 > 1)] = 0
        matrix1 = torch.from_numpy(matrix1).float().to(device)


        matrix2 = PLDistance(ligand_Coord,protein_Coord)
        matrix2[(matrix2 > 0) & (matrix2 <= 1)] = 0
        matrix2[(matrix2 > 1) & (matrix2 <= 2)] = 1
        matrix2[(matrix2 > 2)] = 0
        matrix2 = torch.from_numpy(matrix2).float().to(device)


        matrix3 = PLDistance(ligand_Coord,protein_Coord)
        matrix3[(matrix3 > 0) & (matrix3 <= 2)] = 0
        matrix3[(matrix3 > 2) & (matrix3 <= 3)] = 1
        matrix3[(matrix3 > 3)] = 0
        matrix3 = torch.from_numpy(matrix3).float().to(device)


        matrix4 = PLDistance(ligand_Coord,protein_Coord)
        matrix4[(matrix4 > 0) & (matrix4 <= 3)] = 0
        matrix4[(matrix4 > 3) & (matrix4 <= 4)] = 1
        matrix4[(matrix4 > 4)] = 0
        matrix4 = torch.from_numpy(matrix4).float().to(device)


        matrix5 = PLDistance(ligand_Coord,protein_Coord)
        matrix5[(matrix5 > 0) & (matrix5 <= 4)] = 0
        matrix5[(matrix5 > 4) & (matrix5 <= 5)] = 1
        matrix5[(matrix5 > 5)] = 0
        matrix5 = torch.from_numpy(matrix5).float().to(device)


        matrix6 = PLDistance(ligand_Coord,protein_Coord)
        matrix6[(matrix6 > 0) & (matrix6 <= 5)] = 0
        matrix6[(matrix6 > 5) & (matrix6 <= 6)] = 1
        matrix6[(matrix6 > 6)] = 0
        matrix6 = torch.from_numpy(matrix6).float().to(device)


        matrix7 = PLDistance(ligand_Coord,protein_Coord)
        matrix7[(matrix7 > 0) & (matrix7 <= 6)] = 0
        matrix7[(matrix7 > 6) & (matrix7 <= 7)] = 1
        matrix7[(matrix7 > 7)] = 0
        matrix7 = torch.from_numpy(matrix7).float().to(device)


        matrix8 = PLDistance(ligand_Coord,protein_Coord)
        matrix8[(matrix8 > 0) & (matrix8 <= 7)] = 0
        matrix8[(matrix8 > 7) & (matrix8 <= 8)] = 1
        matrix8[(matrix8 > 8)] = 0
        matrix8 = torch.from_numpy(matrix8).float().to(device)


        matrix9 = PLDistance(ligand_Coord,protein_Coord)
        matrix9[(matrix9 > 0) & (matrix9 <= 8)] = 0
        matrix9[(matrix9 > 8) & (matrix9 <= 9)] = 1
        matrix9[(matrix9 > 9)] = 0
        matrix9 = torch.from_numpy(matrix9).float().to(device)


        matrix10 = PLDistance(ligand_Coord,protein_Coord)
        matrix10[(matrix10 > 0) & (matrix10 <= 9)] = 0
        matrix10[(matrix10 > 9) & (matrix10 <= 10)] = 1
        matrix10[(matrix10 > 10)] = 0
        matrix10 = torch.from_numpy(matrix10).float().to(device)

        return matrix1,matrix2,matrix3,matrix4,matrix5,matrix6,matrix7,matrix8,matrix9,matrix10

    def forward(self, data):
        ligand = data[0]
        protein = data[1]
        ligand_len = data[2]
        protein_len = data[3]


        #version v1.0 demo
        matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8, matrix9, matrix10 = \
            self.weight_matrix(ligand,protein)
        
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

        interaction_map1 = torch.mul(interaction_map, matrix1.to(device)) *(weightblock[0].to(device))
        interaction_map2 = torch.mul(interaction_map, matrix2.to(device)) *(weightblock[1].to(device))
        interaction_map3 = torch.mul(interaction_map, matrix3.to(device)) *(weightblock[2].to(device))
        interaction_map4 = torch.mul(interaction_map, matrix4.to(device)) *(weightblock[3].to(device))
        interaction_map5 = torch.mul(interaction_map, matrix5.to(device)) *(weightblock[4].to(device))
        interaction_map6 = torch.mul(interaction_map, matrix6.to(device)) *(weightblock[5].to(device))
        interaction_map7 = torch.mul(interaction_map, matrix7.to(device)) *(weightblock[6].to(device))
        interaction_map8 = torch.mul(interaction_map, matrix8.to(device)) *(weightblock[7].to(device))
        interaction_map9 = torch.mul(interaction_map, matrix9.to(device)) *(weightblock[8].to(device))
        interaction_map10 = torch.mul(interaction_map, matrix10.to(device)) *(weightblock[9].to(device))


        protein_prime1 = torch.mm(interaction_map1.t(), ligand_features)
        ligand_prime1 = torch.mm(interaction_map1, protein_features)
        protein_prime2 = torch.mm(interaction_map2.t(), ligand_features)
        ligand_prime2 = torch.mm(interaction_map2, protein_features)
        protein_prime3 = torch.mm(interaction_map3.t(), ligand_features)
        ligand_prime3 = torch.mm(interaction_map3, protein_features)
        protein_prime4 = torch.mm(interaction_map4.t(), ligand_features)
        ligand_prime4 = torch.mm(interaction_map4, protein_features)
        protein_prime5 = torch.mm(interaction_map5.t(), ligand_features)
        ligand_prime5 = torch.mm(interaction_map5, protein_features)
        protein_prime6 = torch.mm(interaction_map6.t(), ligand_features)
        ligand_prime6 = torch.mm(interaction_map6, protein_features)
        protein_prime7 = torch.mm(interaction_map7.t(), ligand_features)
        ligand_prime7 = torch.mm(interaction_map7, protein_features)
        protein_prime8 = torch.mm(interaction_map8.t(), ligand_features)
        ligand_prime8 = torch.mm(interaction_map8, protein_features)
        protein_prime9 = torch.mm(interaction_map9.t(), ligand_features)
        ligand_prime9 = torch.mm(interaction_map9, protein_features)
        protein_prime10 = torch.mm(interaction_map10.t(), ligand_features)
        ligand_prime10 = torch.mm(interaction_map10, protein_features)



        # Prediction phase
        ligand_features1 = ligand_prime1
        protein_features1 = protein_prime1
        ligand_features2 = ligand_prime2
        protein_features2 = protein_prime2
        ligand_features3 = ligand_prime3
        protein_features3 = protein_prime3
        ligand_features4 = ligand_prime4
        protein_features4 = protein_prime4
        ligand_features5 = ligand_prime5
        protein_features5 = protein_prime5
        ligand_features6 = ligand_prime6
        protein_features6 = protein_prime6
        ligand_features7 = ligand_prime7
        protein_features7 = protein_prime7
        ligand_features8 = ligand_prime8
        protein_features8 = protein_prime8
        ligand_features9 = ligand_prime9
        protein_features9 = protein_prime9
        ligand_features10 = ligand_prime10
        protein_features10 = protein_prime10

        ligand_features1 = self.set2set_ligand(ligand, ligand_features1)
        protein_features1 = self.set2set_protein(protein, protein_features1)
        ligand_features2 = self.set2set_ligand(ligand, ligand_features2)
        protein_features2 = self.set2set_protein(protein, protein_features2)
        ligand_features3 = self.set2set_ligand(ligand, ligand_features3)
        protein_features3 = self.set2set_protein(protein, protein_features3)
        ligand_features4 = self.set2set_ligand(ligand, ligand_features4)
        protein_features4 = self.set2set_protein(protein, protein_features4)
        ligand_features5 = self.set2set_ligand(ligand, ligand_features5)
        protein_features5 = self.set2set_protein(protein, protein_features5)
        ligand_features6 = self.set2set_ligand(ligand, ligand_features6)
        protein_features6 = self.set2set_protein(protein, protein_features6)
        ligand_features7 = self.set2set_ligand(ligand, ligand_features7)
        protein_features7 = self.set2set_protein(protein, protein_features7)
        ligand_features8 = self.set2set_ligand(ligand, ligand_features8)
        protein_features8 = self.set2set_protein(protein, protein_features8)
        ligand_features9 = self.set2set_ligand(ligand, ligand_features9)
        protein_features9 = self.set2set_protein(protein, protein_features9)
        ligand_features10 = self.set2set_ligand(ligand, ligand_features10)
        protein_features10 = self.set2set_protein(protein, protein_features10)


        final_features1 = torch.cat((ligand_features1, protein_features1), 1)
        final_features2 = torch.cat((ligand_features2, protein_features2), 1)
        final_features3 = torch.cat((ligand_features3, protein_features3), 1)
        final_features4 = torch.cat((ligand_features4, protein_features4), 1)
        final_features5 = torch.cat((ligand_features5, protein_features5), 1)
        final_features6 = torch.cat((ligand_features6, protein_features6), 1)
        final_features7 = torch.cat((ligand_features7, protein_features7), 1)
        final_features8 = torch.cat((ligand_features8, protein_features8), 1)
        final_features9 = torch.cat((ligand_features9, protein_features9), 1)
        final_features10 = torch.cat((ligand_features10, protein_features10), 1)


        final_features =  torch.cat((final_features1, final_features2, final_features3,
                                     final_features4, final_features5, final_features6,
                                     final_features7, final_features8, final_features9,
                                     final_features10), 1)

        predictions = torch.relu(self.fc1(final_features))
        #predictions = self.dp1(predictions)
        predictions = torch.relu(self.fc2(predictions))
       # predictions = self.dp2(predictions)
        predictions = self.fc3(predictions)


        return predictions, ret_interaction_map, weightblock















