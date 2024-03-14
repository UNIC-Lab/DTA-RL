import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.utils import softmax
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from arguments import args

class DTARL(nn.Module):
    def __init__(self, num_layers, user_input_dim, server_input_dim, hidden_dim, alpha=0.02) -> None:
        super(DTARL, self).__init__()

        self.user_encoder = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(alpha)
            )
        
        self.server_encoder = nn.Sequential(
            nn.Linear(server_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(alpha)
        )

        self.convs = nn.ModuleList([HeteroConv(
            {
                ('server', 's2u', 'user'):S2UGNN(hidden_dim, alpha),
                ('user', 'u2u', 'user'):U2UGNN(hidden_dim, alpha)
            },aggr='sum') for _ in range(num_layers)])
        
        self.output_selective_mlp = nn.Sequential(
                                nn.Linear(2*hidden_dim+2, hidden_dim),
                                nn.Sigmoid(),
                                nn.Linear(hidden_dim, 64),
                                nn.Sigmoid(),
                                nn.Linear(64, 1)
        )
        self.offloading_policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 2),
            nn.Softmax()
        )


        self.att_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        x_dict['user'] = self.user_encoder(x_dict['user'])
        x_dict['server'] = self.server_encoder(x_dict['server'])

        # user_feats = []
        for conv in self.convs:
            user_feat = conv(x_dict, edge_index_dict, edge_attr_dict)['user']
            x_dict['user'] = user_feat
        user_node_feats = user_feat

        user_idx = edge_index_dict['user', 'u2s', 'server'][0]
        server_idx = edge_index_dict['user', 'u2s', 'server'][1]
        # u2u_src = edge_index_dict['user', 'u2u', 'user'][0]
        # u2u_dst = edge_index_dict['user', 'u2u', 'user'][1] # 邻居节点的索引
        # neighbor_h = torch.index_add(user_node_feats, dim=0, index=u2u_src, source=user_node_feats[u2u_dst])  # 聚合邻居节点的特征
        # user_node_feats = torch.cat([user_node_feats, neighbor_h-user_node_feats], dim=-1)
        server_node_feats = x_dict['server'].clone()

        # 每个用户是否卸载概率
        user_offloading_probs = self.offloading_policy_mlp(user_node_feats)

        # 用户选择的卸载对象
        catted = torch.cat([user_node_feats[user_idx], user_offloading_probs[user_idx], server_node_feats[server_idx]], dim=-1)      # 用户节点特征加上用户是否选择卸载，再与服务器的节点特征聚合
        
        att_weights = self.output_selective_mlp(catted).squeeze()       # 用户与服务器之间的注意力
        server_selection_scheme = softmax(att_weights, user_idx)   # 每条边被选中为用户的概率
        
        return user_offloading_probs, server_selection_scheme


class U2UGNN(MessagePassing):
    def __init__(self, hidden_dim, alpha, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super(U2UGNN, self).__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU(),
        )
        self.update_lin = nn.Linear(2*hidden_dim, hidden_dim)
        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.LeakyReLU(alpha),
            nn.Linear(hidden_dim, 1)
        )

        self.relation_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim+1, hidden_dim),
            nn.ReLU()
        )
        
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_index, edge_attr) -> Tensor:
        # 消息传播机制
        # message_mlp计算用户到server的信息传递
        tmp = torch.cat([x_i, edge_attr], dim=1)
        # 计算注意力
        rela_feat = torch.cat([self.Wq(x_i), self.Wr(x_j)], dim=1)
        att_weight = self.att(rela_feat)
        
        att_weight = att_weight = softmax(att_weight, index=edge_index[1], dim=0)
        outputs = self.message_mlp(tmp)
        outputs = att_weight*outputs

        # 将注意力特征与边特征结合得到user到server的关系特征
        tmp = torch.cat([x_j, outputs, edge_attr], dim=-1)       # user的特征，server到user的关系特征与边特征
        self.message_attr = self.relation_mlp(tmp)

        return outputs
    def update(self, aggr_out, x) -> Tensor:

        output = self.update_lin(torch.column_stack([aggr_out, x]))
        
        return output

class S2UGNN(MessagePassing):
    def __init__(self, hidden_dim, alpha, aggr: Optional[str] = "add", flow: str = "source_to_target", node_dim: int = -2, decomposed_layers: int = 1):
        super(S2UGNN, self).__init__()
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim+1, hidden_dim),
            nn.ReLU()
        )
        self.update_lin = nn.Linear(2*hidden_dim, hidden_dim)
        self.att = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.LeakyReLU(alpha),
            nn.Linear(hidden_dim, 1)
        )

        self.relation_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim+1, hidden_dim),
            nn.ReLU()
        )
        
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        x_src = x[0]
        x_dst = x[1]
        return self.propagate(x=(x_src, x_dst), edge_index=edge_index, edge_attr=edge_attr)
    def message(self, x_i, x_j, edge_index, edge_attr) -> Tensor:
        # 消息传播机制

        # message_mlp计算server到user的信息传递
        tmp = torch.cat([x_i, edge_attr], dim=1)
        # 计算注意力
        rela_feat = torch.cat([self.Wq(x_i), self.Wr(x_j)], dim=1)
        att_weight = self.att(rela_feat)
        
        att_weight = softmax(att_weight, index=edge_index[1], dim=0)
        outputs = self.message_mlp(tmp)
        outputs = att_weight*outputs

        # 将注意力特征与边特征结合得到user到server的关系特征
        tmp = torch.cat([x_j, outputs, edge_attr], dim=-1)       # user的特征，server到user的关系特征与边特征
        self.message_attr = self.relation_mlp(tmp)

        return outputs
    def update(self, aggr_out, x) -> Tensor:
        output = F.relu(self.update_lin(torch.column_stack([aggr_out, x[1]])))
        return output


class FixedRL(nn.Module):
    def __init__(self, hidden_dim, user_dim, server_dim, max_user_num, max_server_num) -> None:
        super(FixedRL, self).__init__()
        
        self.max_user_num = max_user_num
        self.max_server_num = max_server_num
        self.user_dim = user_dim
        self.server_dim = server_dim
        
        self.offloading_mlp = nn.Sequential(
                        nn.Linear(max_user_num*user_dim+max_server_num*server_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Sigmoid(),
                        nn.Linear(hidden_dim, 2),
                        nn.Softmax(-1)
        )
        self.selection_mlp = nn.Sequential(
                        nn.Linear(max_user_num*user_dim+max_server_num*server_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Sigmoid(),
                        nn.Linear(hidden_dim, max_server_num)
        )
        
    def forward(self, user_batch, server_batch, x_dict, edge_index_dict):
        src_u2s = edge_index_dict['user', 'u2s', 'server'][0]
        dst_u2s = edge_index_dict['user', 'u2s', 'server'][1]
        src_u2u = edge_index_dict['user', 'u2u', 'user'][0]
        dst_u2u = edge_index_dict['user', 'u2u', 'user'][1]
        
        batch_num = user_batch.max()+1
        server_num = x_dict['server'].shape[0]//batch_num

        server_obs_tensor = torch.zeros((x_dict['user'].shape[0], batch_num*self.max_server_num*self.server_dim), device=args.device)
        user_obs_tensor = torch.zeros((x_dict['user'].shape[0], batch_num*self.max_user_num*self.user_dim), device=args.device)
        
        selected_server_features = x_dict['server'].index_select(0, dst_u2s)
        dst_u2s_1d = dst_u2s.view(-1, 1)
        offsets = dst_u2s_1d * self.server_dim
        server_index = torch.arange(self.server_dim, device=args.device).expand_as(selected_server_features)
        src_expanded = src_u2s.view(-1, 1).expand_as(selected_server_features)
        server_obs_tensor[src_expanded.reshape(-1), (offsets + server_index).reshape(-1)] = selected_server_features.reshape(-1)
        
        selected_user_features = x_dict['user'].index_select(0, dst_u2u)
        dst_u2u_1d = dst_u2u.view(-1, 1)
        offsets = dst_u2u_1d * self.user_dim
        user_index = torch.arange(self.user_dim, device=args.device).expand_as(selected_user_features)
        src_expanded = src_u2u.view(-1, 1).expand_as(selected_user_features)
        user_obs_tensor[src_expanded.reshape(-1), (offsets + user_index).reshape(-1)] = selected_user_features.reshape(-1)


        user_batch_extended = torch.arange(batch_num, device=args.device).unsqueeze(-1).repeat(1, self.max_user_num*self.user_dim).reshape(-1)
        server_batch_extended = torch.arange(batch_num, device=args.device).unsqueeze(-1).repeat(1, self.max_server_num*self.server_dim).reshape(-1)
        
        user_mask = user_batch.view(-1, 1)==user_batch_extended
        user_obs_tensor = user_obs_tensor[user_mask].view(batch_num, -1, self.max_user_num*self.user_dim)
        server_mask = user_batch.view(-1, 1)==server_batch_extended
        server_obs_tensor = server_obs_tensor[server_mask].view(batch_num, -1, self.max_server_num*self.server_dim)

        feats = torch.cat([server_obs_tensor, user_obs_tensor], dim=-1)
        offloading_probs = self.offloading_mlp(feats)
        selection_probs = self.selection_mlp(feats)

        # 去除无法连接的服务器
        server_batch = torch.arange(batch_num, device=args.device).unsqueeze(-1).repeat(1, server_num).reshape(-1)
        mask = user_batch.view(-1, 1)==server_batch
        non_probs = torch.full((batch_num*selection_probs.shape[1], x_dict['server'].shape[0]), -torch.inf, device=args.device)  # (batch*user_num, max_server_num)
        non_probs[src_u2s, dst_u2s] = 0     # 有连接的边
        non_probs = non_probs[mask].view(batch_num, -1, server_num)
        non_probs = torch.cat([non_probs, torch.full((selection_probs.shape[0], selection_probs.shape[1], selection_probs.shape[-1]-non_probs.shape[-1]), -torch.inf, device=args.device)], dim=-1)
        selection_probs = selection_probs + non_probs
        
        # softmax
        selection_probs = torch.softmax(selection_probs, dim=-1)

        return offloading_probs, selection_probs
      