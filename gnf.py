import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.distributions import Normal
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid
from torch.optim import Adam
from torch_geometric.utils import train_test_split_edges
import argparse

# Definition of Encoder
class Encoder(nn.Module):

    def __init__(self, input_channel):
        super(Encoder,self).__init__()
        self.encoder1 = GATConv(input_channel,50)
        self.encoder2 = GATConv(50,40)

    def forward(self,x):
        x, edge_index = x.x, x.train_pos_edge_index
        x = F.relu(self.encoder1(x,edge_index))
        x = F.relu(self.encoder2(x,edge_index))
        return x, edge_index

# Definition of Encoder
class Decoder(nn.Module):

    def __init__(self, input_channel):
        super(Decoder,self).__init__()
        self.decoder1 = GATConv(40,50)
        self.decoder2 = GATConv(50,input_channel)
        self.decoder3 = InnerProductDecoder()

    def forward(self,x,edge_index):
        x = F.relu(self.decoder1(x,edge_index))
        x = F.relu(self.decoder2(x,edge_index))
        adj_pred = self.decoder3.forward_all(x, sigmoid=True)
        return adj_pred, edge_index

# Definition of GNF
class GNF(nn.Module):

    def __init__(self):
        super(GNF,self).__init__()
        self.F1 = GATConv(20,20)
        self.F2 = GATConv(20,20)
        self.G1 = GATConv(20,20)
        self.G2 = GATConv(20,20)

    def forward(self,x,edge_index):
        x1,x2 = x[:,:20], x[:,20:]
        x1,x2,s1 = self.f_a(x1,x2,edge_index)
        x2,x1,s2 = self.f_b(x2,x1,edge_index)
        # Calculate Jacobian
        log_det_xz = torch.sum(s1,axis=1) + torch.sum(s2,axis=1)
        return x1, x2, log_det_xz

    def f_a(self,x1,x2,edge_index):
        s_x = self.F1(x1,edge_index)
        t_x = self.F2(x1,edge_index)
        x1 = x2 * torch.exp(s_x) + t_x
        return x1,x2,s_x

    def f_b(self,x1,x2,edge_index):
        s_x = self.G1(x1,edge_index)
        t_x = self.G2(x1,edge_index)
        x1 = x2 * torch.exp(s_x) + t_x
        return x1,x2,s_x

    def inverse_b(self,z1,z2,edge_index):
        s_x = self.G1(z1,edge_index)
        t_x = self.G2(z1,edge_index)
        z2 = (z2 - t_x) * torch.exp(-s_x)
        return z1,z2,s_x

    def inverse_a(self,z1,z2,edge_index):
        s_x = self.F1(z1,edge_index)
        t_x = self.F2(z1,edge_index)
        z2 = (z2 - t_x) * torch.exp(-s_x)
        return z1,z2,s_x

    def inverse(self,z,edge_index):
        z1,z2 = z[:,:20], z[:,20:]
        z1,z2,s1 = self.inverse_b(z1,z2,edge_index)
        z2,z1,s2 = self.inverse_a(z2,z1,edge_index)
        # Calculate Jacobian
        log_det_zx = torch.sum(-s1,axis=1) + torch.sum(-s2,axis=1)
        return z1, z2, log_det_zx


    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index, encoder, decoder):

        z,_ = encoder(x)
        
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index, decoder)
        return roc_auc_score, average_precision_score

    from torch import Tensor
    from typing import Optional, Tuple
    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor,decoder) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred =  decoder.decoder3.forward(z, pos_edge_index)
        neg_pred = decoder.decoder3.forward(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)



if __name__ == "__main__":

    args =argparse.ArgumentParser()
    args.add_argument('--lr', type=float, default=0.01 )
    args.add_argument('--dataset', type=str, default="Cora")
    args.add_argument('--epochs', type=int, default=10)
    args = args.parse_args()


    if args.dataset == "Cora":
        input_channel = 1433
    elif args.dataset == "Citeseer":
        input_channel = 3703
    elif args.dataset == "PubMed":
        input_channel = 500

    torch.manual_seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid("datasets", args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    all_edge_index = data.edge_index
    data = train_test_split_edges(data, 0.05, 0.01)
        

    encoder = Encoder(input_channel)
    decoder = Decoder(input_channel)
    gnf = GNF()

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(gnf.parameters())
    optimizer = Adam(params , lr=args.lr)
    max_epoch = args.epochs


    for epoch in range(max_epoch):

        data_, edge_index = encoder(data)
        # Forward

        z1,z2,log_det_xz = gnf.forward(data_,edge_index)
        print(log_det_xz)


        log_det_xz = log_det_xz.unsqueeze(0)
        log_det_xz = log_det_xz.mm(torch.ones(log_det_xz.t().size()))
        #self.log_det_xz.append(log_det_xz)

        # Sample
        z = torch.cat((z1,z2),dim=1)
        n = Normal(z, torch.ones(z.size())*0.3)
        z = n.sample() # bug needs to be fixed 
        print(z.shape)
        # Inverse
        z1,z2, log_det_zx = gnf.inverse(z,edge_index)
        log_det_zx = log_det_zx.unsqueeze(0)
        log_det_zx = log_det_zx.mm(torch.ones(log_det_zx.t().size()))
        # Decoder
        z = torch.cat((z1,z2),dim=1)


        y, _ = decoder(z,edge_index)

        pos_loss = -torch.log(
            y).mean() 
        

        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), data.train_pos_edge_index.size(1))

        neg_loss = -torch.log(1 - y + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        roc_auc, ap = gnf.single_test(data,
                                        data.train_pos_edge_index,
                                        data.test_pos_edge_index,
                                        data.test_neg_edge_index, encoder, decoder)
        print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))

    print('Done')
