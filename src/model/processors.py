import torch
import torch.nn as nn
from torch_geometric.nn import aggr
from torch_geometric.utils import degree, scatter, dropout


class BipartiteMPNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.msg_x_to_y = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        self.update_x = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        self.global_pool = aggr.MinAggregation()

    def forward(self, h_x, h_x_degree, edge_index, x_mask, y_mask, edge_mask, batch_index_x, batch_index_y, batch_size, eps=False):
        # Get edge_index 
        edge_index = edge_index[:, edge_mask]  # edge_mask[e] is False if the edge has been deleted

        # Pass messages from x (RHS) to y (LHS) 
        x_degree = self.msg_x_to_y(torch.cat((h_x, h_x_degree), dim=-1)) 
        msg_x_to_y = x_degree[edge_index[1]]
        next_y = scatter(msg_x_to_y, index=edge_index[0], dim=0, dim_size=y_mask.shape[0], reduce='min')  

        if eps: 
            # Aggregate messages to compute epsilon 
            next_y_masked = next_y[y_mask.squeeze(-1)]
            batch_index_y_masked = batch_index_y[y_mask.squeeze(-1)]
            next_eps = scatter(next_y_masked, index=batch_index_y_masked, dim=0, dim_size=batch_size, reduce='min')

            # Repeat eps to send back to x 
            next_eps_y = next_eps[batch_index_y] 
            eps_feat = next_eps_y[edge_index[0]]

            # Pass the aggregated message back to LHS nodes
            msg_y_to_x = scatter(eps_feat, index=edge_index[1], dim=0, dim_size=x_mask.shape[0], reduce='sum')  
        else:
            msg_y_to_x = next_y[edge_index[0]]
            msg_y_to_x = scatter(msg_y_to_x, index=edge_index[1], dim=0, dim_size=x_mask.shape[0], reduce='sum')  
            next_eps = None

        # Update x (RHS) 
        next_x = self.update_x(torch.cat((h_x, msg_y_to_x), dim=-1))  

        assert torch.isnan(next_x).sum() == 0
        assert torch.isnan(next_y).sum() == 0
        if eps:
            assert torch.isnan(next_eps).sum() == 0

        return next_x, next_y, next_eps 

