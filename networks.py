import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn.pool import SAGPooling, EdgePooling, ClusterPooling, ASAPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F





class SAGNet(torch.nn.Module):
    def __init__(self,args):
        super(SAGNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch,_ ,_ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,_ ,_ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch,_ ,_ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

class TopKNet(SAGNet):
    def __init__(self, args):
        super(TopKNet, self).__init__(args)
        self.pool1 = TopKPooling(self.nhid, ratio=self.pooling_ratio)
        self.pool2 = TopKPooling(self.nhid, ratio=self.pooling_ratio)
        self.pool3 = TopKPooling(self.nhid, ratio=self.pooling_ratio)
    
class EdgeNet(SAGNet):
    def __init__(self, args):
        super(EdgeNet, self).__init__(args)
        self.pool1 = EdgePooling(self.nhid)
        self.pool2 = EdgePooling(self.nhid)
        self.pool3 = EdgePooling(self.nhid)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch, _ = self.pool1(x, edge_index, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, batch, _ = self.pool2(x, edge_index, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, batch, _ = self.pool3(x, edge_index, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
    
class ClusterNet(EdgeNet):
    def __init__(self, args):
        super(ClusterNet, self).__init__(args)
        self.pool1 = ClusterPooling(self.nhid)
        self.pool2 = ClusterPooling(self.nhid)
        self.pool3 = ClusterPooling(self.nhid)

class ASANet(SAGNet):
    def __init__(self, args):
        super(ASANet, self).__init__(args)
        self.pool1 = ASAPooling(self.nhid, self.pooling_ratio, gnn_intra_cluster=GCNConv(self.nhid, self.nhid))
        self.pool2 = ASAPooling(self.nhid, self.pooling_ratio, gnn_intra_cluster=GCNConv(self.nhid, self.nhid))
        self.pool3 = ASAPooling(self.nhid, self.pooling_ratio, gnn_intra_cluster=GCNConv(self.nhid, self.nhid))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch,_ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,_ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch,_ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x