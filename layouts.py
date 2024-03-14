import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData


    
def compute_path_losses(args, distances):
    # f_min = args.carrier_f_start
    # f_max = args.carrier_f_end
    # f_gap = f_max-f_min
    # # carrier_lam = args.carrier_lam          # len N verctor
    # server_range = np.arange(distances.shape[1])
    # carrier_f = f_min + server_range * f_gap
    # carrier_lam = 2.998e8 / carrier_f
    # signal_cof = args.signal_cof
    # path_losses = (signal_cof * carrier_lam) / (distances**2.75)
    path_losses = args.Gr/(args.L*(distances**2.75))
    return path_losses



def build_graph(user_feat, server_feat, user_coords, server_coords, task_size, p_max, s2u_index, u2s_index, u2u_index, s2s_index, mask_u2s, mask_u2u, u2u_distance_normed, u2s_path_loss, u2s_path_loss_normed, args, env_len):
    
    user_ones = np.ones((len(user_feat), 1))
    server_ones = np.ones((len(server_feat), 1))
    user_feat = np.concatenate((user_feat, user_ones), axis=1)  # 特征补1
    server_feat = np.concatenate((server_feat, server_ones), axis=1)    # 特征补1
    user_feat = torch.tensor(user_feat, dtype=torch.float).to(args.device)
    server_feat = torch.tensor(server_feat, dtype=torch.float).to(args.device)
    u2s_path_loss = torch.tensor(u2s_path_loss, dtype=torch.float).to(args.device)

    user_coords = torch.tensor(user_coords, dtype=torch.float).to(args.device)
    server_coords = torch.tensor(server_coords, dtype=torch.float).to(args.device)
    
    task_size = torch.tensor(task_size, dtype=torch.float, device=args.device)
    p_max = torch.tensor(p_max, dtype=torch.float, device=args.device)
    s2u_attr = u2s_path_loss.T[np.where(mask_u2s.T!=0)].unsqueeze(-1)*1e10
    u2s_attr = u2s_path_loss[np.where(mask_u2s!=0)].unsqueeze(-1)*1e10
    
    u2u_distance_attr = u2u_distance_normed[mask_u2u!=0][:, np.newaxis]
    u2u_attr = torch.tensor(u2u_distance_attr, dtype=torch.float).to(args.device)     # 去除自环



    s2u_idx = torch.tensor(s2u_index, dtype=torch.long).to(args.device)
    u2s_index = torch.tensor(u2s_index, dtype=torch.long).to(args.device)
    u2u_idx = torch.tensor(u2u_index, dtype=torch.long).to(args.device)
    s2s_idx = torch.tensor(s2s_index, dtype=torch.long).to(args.device)

    data = HeteroData().to(args.device)



    data['user'].x = user_feat      # user的位置，user要卸载的task的大小
    data['server'].x = server_feat
    data['user'].coords = user_coords
    data['user'].p_max = p_max
    data['user'].task_size = task_size
    data['server'].coords = server_coords
    data['server'].comp_max = server_feat[:, 2]*args.user_comp_max
    data['user', 'u2u', 'user'].edge_index = u2u_idx
    data['server', 's2u', 'user'].edge_index = s2u_idx
    data['user', 'u2s', 'server'].edge_index = u2s_index

    data['user', 'u2s', 'server'].path_loss = u2s_path_loss[np.where(mask_u2s!=0)].reshape((-1, 1))

    data['server', 's2s', 'server'].edge_index = s2s_idx
    data['server', 's2u', 'user'].edge_attr = s2u_attr
    data['user', 'u2s', 'server'].edge_attr = u2s_attr
    data['user', 'u2u', 'user'].edge_attr = u2u_attr

    return data



def generate_layouts(user_nums, server_nums, args):
    
    graphs = []
    
    user_generate_task_probs = []
    for idx in range(len(user_nums)):
        
        user_num_idx = user_nums[idx]   # 当前场景用户的个数
        server_num_idx = server_nums[idx]   # 当前场景服务器的个数
        env_len = np.sqrt(user_nums[idx])*500   # 每1k平方公里2个服务器

        # 归一化位置, tasksize, computing resource
        user_corrds = np.random.random([user_nums[idx], 2])
        user_corrds = user_corrds * env_len   # user的随机位置
        user_idx_feat = user_corrds

        server_corrds = np.random.random([server_nums[idx], 2])   
        server_corrds = server_corrds * env_len   # 服务器位置
        server_idx_feat = server_corrds

        user_task_prob = np.random.uniform(0.3, 0.7, (user_nums[idx]))
        # user_task_prob = np.ones((user_num_idx))
        user_generate_task_probs.append(user_task_prob)

        # 保证每个用户都能连接至少一个服务器
        for i in range(user_nums[idx]):
            while min(np.sum((server_corrds - user_corrds[i])**2, axis=1)) > args.threshold**2:
                # If not, reposition the user
                user_corrds[i] = np.random.rand(2) * env_len
        
        
        # 生成初始的任务大小和任务的计算消耗
        user_idx_tasksize = np.random.randint(args.task_min, args.task_max, size=(user_nums[idx], 1))     # 初始任务的大小
        user_idx_taskcomp = np.round(np.random.uniform(args.task_comp_min, args.task_comp_max, size=(user_nums[idx], 1)), 5)     # 初始任务的每bit需要的计算量
        user_idx_pmax = np.random.random((user_nums[idx], 1))*(args.p_max-args.p_min)+args.p_min        # 用户的最大传输功率，随机设置
        user_idx_comp_frez = np.round(np.random.uniform(args.user_comp_min, args.user_comp_max, size=(user_nums[idx], 1)), 7)    # 用户的计算速率，随机设置
        user_idx_feat = np.concatenate([user_idx_feat, user_idx_pmax, user_idx_comp_frez, user_idx_tasksize, user_idx_taskcomp, np.zeros_like(user_idx_comp_frez)], axis=-1)      # 用户特征，最后一个特征为队列排队时间

        # server的计算资源
        server_idx_comp = np.round(np.random.uniform(args.server_comp_min, args.server_comp_max, size=(server_nums[idx], 1)), 7)   # 每个服务器的计算资源，即计算速率

        
        server_idx_feat = np.concatenate([server_idx_feat, server_idx_comp, np.zeros_like(server_idx_comp)], axis=-1)    # 服务器特征，最后一个特征为队列排队时间


        u2u_distances_idx = np.sqrt(np.sum((user_corrds[:, np.newaxis] - user_corrds) ** 2, axis=2))        # 用户与用户之间的距离
        u2s_distances_idx = np.sqrt(np.sum((user_corrds[:, np.newaxis] - server_corrds) ** 2, axis=2))      # 用户与服务器之间的距离
        u2s_path_loss = compute_path_losses(args, u2s_distances_idx)    # 用户与服务器之间的path loss

        mask_u2s = np.repeat(np.ones(server_num_idx)[np.newaxis, :], repeats=user_num_idx, axis=0)      # adj matrix of user to server
        mask_u2s[np.where(u2s_distances_idx>args.threshold)] = 0    # 去除超过阈值的边

        mask_u2u = np.repeat(np.ones(user_num_idx)[np.newaxis, :], repeats=user_num_idx, axis=0)    # adj matrix of user to user
        mask_u2u = mask_u2u-np.eye(user_num_idx)    # 去除自环
        mask_u2u[np.where(u2u_distances_idx>args.threshold)] = 0    # 去除超过阈值的边

        mask_s2s = np.repeat(np.ones(server_num_idx)[np.newaxis, :], repeats=server_num_idx, axis=0)    # adj matrix of server to server
        mask_s2s = mask_s2s-np.eye(server_num_idx)  # 去除自环



        # edge_index of users to users
        index_u2u_src, index_u2u_dst = np.nonzero(mask_u2u)
        u2u_index = np.concatenate([index_u2u_src[np.newaxis, :], index_u2u_dst[np.newaxis, :]], axis=0)    

        # edge_index of servers to users
        index_s2u_dst, index_s2u_src = np.nonzero(mask_u2s)
        s2u_index = np.concatenate([index_s2u_src[np.newaxis, :], index_s2u_dst[np.newaxis, :]], axis=0)

        index_u2s_src, index_u2s_dst = np.nonzero(mask_u2s)
        u2s_index = np.concatenate([index_u2s_src[np.newaxis, :], index_u2s_dst[np.newaxis, :]], axis=0)

        index_s2s_src, index_s2s_dst = np.nonzero(mask_s2s)
        s2s_index = np.concatenate([index_s2s_src[np.newaxis, :], index_s2s_dst[np.newaxis, :]], axis=0)

        
        graph = build_graph(user_idx_feat, server_idx_feat, user_corrds, server_corrds, user_idx_tasksize, user_idx_pmax, s2u_index, u2s_index, u2u_index, s2s_index, mask_u2s, mask_u2u, u2u_distances_idx, u2s_path_loss, u2s_path_loss, args, env_len)
        graphs.append(graph)

        
    loader = DataLoader(graphs, batch_size=args.train_layouts)
    user_generate_task_probs = np.concatenate(user_generate_task_probs)
    return loader, user_generate_task_probs



