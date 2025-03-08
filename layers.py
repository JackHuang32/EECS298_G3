from torch_geometric.nn import GCNConv
# from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn.pool.topk_pool import TopKPooling
from torch_geometric.utils import scatter 
from torch_geometric.utils import subgraph
from torch.nn import Parameter
import torch

def grouped_argsort(score, ratio, batch):
    sorted_idx = torch.argsort(score, descending=True)  # 先排序
    mask = torch.zeros_like(score, dtype=torch.bool)  # 創建 mask
    
    for b in torch.unique(batch):  # 遍歷所有 batch
        batch_mask = (batch == b)  # 找到 batch b 的所有節點
        num_nodes = batch_mask.sum().item()  # 計算這個 batch 的節點數
        num_keep = int(num_nodes * ratio)  # 計算要保留的節點數

        batch_sorted = sorted_idx[batch_mask][:num_keep]  # 取出該 batch 內最好的節點
        mask[batch_sorted] = True  # 標記這些節點要保留

    return torch.where(mask)[0]  # 回傳被選中的索引

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        print(f'batch: {batch}')
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = grouped_argsort(score, self.ratio, batch)
        # perm = torch.argsort(score, descending=True)[:int(self.ratio * score.size(0))]
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        # edge_index, edge_attr = filter_adj(
            # edge_index, edge_attr, perm, num_nodes=score.size(0))
        edge_index, edge_attr = subgraph(perm, edge_index, edge_attr, relabel_nodes=True, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm