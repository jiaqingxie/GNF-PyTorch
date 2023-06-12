import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch.distributions import Normal
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid
from torch.optim import Adam
from torch_geometric.utils import train_test_split_edges
import argparse
from torch import Tensor
from typing import Optional, Tuple

# Definition of Encoder
class Encoder(nn.Module):

    def __init__(self, input_channel):
        super(Encoder,self).__init__()
        self.encoder1 = GATConv(input_channel,32)
        # self.encoder2 = GATConv(32,32)

    def forward(self,x):
        x, edge_index = x.x, x.train_pos_edge_index
        x = F.relu(self.encoder1(x,edge_index))
        # x = F.relu(self.encoder2(x,edge_index))
        return x, edge_index

# Definition of Encoder
class Decoder(nn.Module):

    def __init__(self, input_channel):
        super(Decoder,self).__init__()
        # self.decoder1 = GCNConv(40,50)
        # self.decoder2 = GCNConv(50,input_channel)
        self.decoder3 = InnerProductDecoder()

    def forward(self,x,edge_index):
        # x = F.relu(self.decoder1(x,edge_index))
        # x = F.relu(self.decoder2(x,edge_index))
        adj_pred = self.decoder3(x, edge_index, sigmoid=True) + 1e-15
        return adj_pred, edge_index

# Definition of GVAE
class GVAE(nn.Module):

    def __init__(self):
        super(GVAE,self).__init__()

        self.gcn_mu = GATConv(32,16)
        self.gcn_logvar = GATConv(32,16)
        self.gcn_mu1 = GATConv(40,40)
        self.gcn_logvar1 = GATConv(40,40)


    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu
        
    
    def kl_loss(self, mu: Tensor = None,
                logstd: Tensor = None) -> Tensor:

        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index, encoder, decoder):
        with torch.no_grad():
            z,_ = encoder(x)
            mu = gnf.gcn_mu(z, edge_index)
            logstd = gnf.gcn_logvar(z, edge_index)
            logstd = logstd.clamp(max=10) 

            z = gnf.reparametrize(mu, logstd)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index, decoder)
        return roc_auc_score, average_precision_score


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

        pos_pred,_ =  decoder(z, pos_edge_index)
        neg_pred,_ = decoder(z, neg_edge_index)
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

    torch.manual_seed(114514)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid("datasets", args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    all_edge_index = data.edge_index
    data = train_test_split_edges(data, 0.02, 0.01)
        

    encoder = Encoder(input_channel).to(device)
    decoder = Decoder(input_channel).to(device)
    gnf = GVAE().to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(gnf.parameters())
    optimizer = Adam(params , lr=args.lr)
    max_epoch = args.epochs


    for epoch in range(max_epoch):

        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        data_, edge_index = encoder(data)

        mu = gnf.gcn_mu(data_, edge_index)
        logstd = gnf.gcn_logvar(data_, edge_index)
        logstd = logstd.clamp(max=0.1) 
        z = gnf.reparametrize(mu, logstd)

        kl_loss = 1 / data_.size(0) * gnf.kl_loss(mu, logstd)

        y,_ = decoder(z, edge_index =  edge_index) 

        pos_loss = -torch.log(
            y).mean() 
        

        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), edge_index.size(1))

        a,_ = decoder(z, neg_edge_index)

        neg_loss = -torch.log(1 - a + 1e-15).mean()

        loss = neg_loss + pos_loss + kl_loss
        loss.backward()
        optimizer.step()

        roc_auc, ap = gnf.single_test(data,
                                        data.train_pos_edge_index,
                                        data.test_pos_edge_index,
                                        data.test_neg_edge_index, encoder, decoder)
        print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))

    print('Done')
