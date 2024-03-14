import torch
from off_loading_rl import DTARL, FixedRL
from arguments import args
from layouts import generate_layouts
from runner import Runner
import numpy as np
import random


class Env:
    def __init__(self, train_loader, user_generate_task_probs, user_num, server_num, args) -> None:
        self.origin_graphs = next(iter(train_loader)).to(args.device)  # 整个聚合的大图
        self.graphs = self.origin_graphs.clone()
        self.user_generate_tasks_probs = user_generate_task_probs
        self.user_num = user_num
        self.server_num = server_num
        # 队列长度
        self.user_queue_time = self.graphs['user'].x[:, -2].reshape(-1, user_num)
        self.server_queue_time = self.graphs['server'].x[:, -2].reshape(-1, server_num)
        pathloss_edges = self.graphs['user', 'u2s', 'server'].path_loss
        self.pathloss = torch.zeros(self.graphs['user'].x.shape[0], self.graphs['server'].x.shape[0], device=args.device)
        self.pathloss[self.graphs['user', 'u2s', 'server'].edge_index[0], self.graphs['user', 'u2s', 'server'].edge_index[1]] = pathloss_edges.squeeze()
        user_batch = self.graphs['user'].batch
        server_batch = self.graphs['server'].batch
        mask = user_batch.view(-1, 1)==server_batch
        self.pathloss = self.pathloss[mask].view(-1, user_num, server_num)
        self.args = args
    
    def restart_(self):
        self.graphs = self.origin_graphs.clone()
        self.user_queue_time = self.graphs['user'].x[:, -2].reshape(-1, self.user_num)
        self.server_queue_time = self.graphs['server'].x[:, -2].reshape(-1, self.server_num)

    def generate_tasks(self, num_steps):
        tasks = []
        for _ in range(num_steps):
            _, task_sizes, task_resources = self.generate_task_one_step()
            task_sizes = torch.FloatTensor(task_sizes, device='cpu')
            task_resources = torch.FloatTensor(task_resources, device='cpu')
            tasks.append({'size': task_sizes, 'comp': task_resources})
        return tasks
    def generate_task_one_step(self):
        # 生成一步的任务
        arrival_flags = []
        user_idxs = []
        task_sizes_list = []
        task_resources_list = []
        for user_idx in range(self.graphs['user'].x.shape[0]):
            if random.random() < self.user_generate_tasks_probs[user_idx]:     # 不同用户的参数可以不一样
                arrival_flags.append(1)
                task_size = round(random.uniform(self.args.task_min, self.args.task_max), 5)       # 任务的大小
                task_resources = random.randint(self.args.task_comp_min, self.args.task_comp_max)     # 任务的每bit计算需求
            else:       # 如果不到达，任务大小和计算需求就为0
                arrival_flags.append(0)
                task_size = 0       # 任务的大小
                task_resources = 0     # 任务的每bit计算需求
            task_sizes_list.append(task_size)
            task_resources_list.append(task_resources)
        return arrival_flags, task_sizes_list, task_resources_list
    

    def state_update(self, local_comp_time, comp_time_each_user, server_selection, next_task):
        # 更新服务器的排队时间
        server_queue_time = torch.scatter_add(self.server_queue_time, 1, server_selection, comp_time_each_user)
        # server_compute_time = torch.zeros(self.server_queue_time.shape[0], device=self.args.device)
        # server_compute_time.scatter_add_(0, server_selection, comp_time_each_user)
        server_queue_time = server_queue_time - self.args.delta_t
        server_queue_time[server_queue_time<0] = 0
        self.server_queue_time = server_queue_time
        # 更新用户的排队时间
        user_queue_time = self.user_queue_time + local_comp_time - self.args.delta_t
        user_queue_time[user_queue_time<0] = 0
        self.user_queue_time = user_queue_time
        # 更新grap
        self.graphs = self.origin_graphs.clone()
        # 更新用户的任务大小以及计算需求
        self.graphs['user'].x[:, 4] = next_task['size']
        self.graphs['user'].x[:, 5] = next_task['comp']
        self.graphs['user'].x[:, -2] = self.user_queue_time.view(-1)
        self.graphs['server'].x[:, -2] = self.server_queue_time.view(-1)
        return self.graphs
    
    def selective_step(self, off_actions, server_selection, next_task):
        '''
            server_selection: [batch, user_num]
            task_size: [batch, user_num]
            trans_power: [batch, user_num]
            off_actions: [batch, user_num]
        '''
        TINY=1e-20
        # off_actions = off_actions.reshape((-1, ))
        # server_selection = server_selection.reshape((-1, ))
        user_comp_frez = (self.graphs['user'].x[:, 3] * self.args.comp_cof).reshape(-1, self.user_num)
        server_comp_frez = (self.graphs['server'].x[:, 2] * self.args.comp_cof).reshape(-1, self.server_num)
        task_comp_eff = self.graphs['user'].x[:, 5].reshape(-1, self.user_num)     # 任务计算速率
        task_size = (self.graphs['user'].x[:, 4] * self.args.size_cof).reshape(-1, self.user_num)     # 任务大小
        
        # off_actions为0时，不卸载，不计算传输速率
        # 计算本地卸载时间
        task_size_copy = task_size.clone()
        task_size_copy[off_actions==1] = 0
        local_comp_time = torch.div(task_size_copy*task_comp_eff, user_comp_frez)     # 任务不卸载用户的本地计算时间
        local_queue_time = self.user_queue_time.clone()     # 任务不卸载用户的本地排队时间
        local_queue_time[off_actions==1] = 0        # 任务卸载时本地排队时间为0
        # off_actions为1时，卸载，计算传输速率
        
        trans_rate = self.transmission_rate(off_actions, server_selection)     # 根据服务器选择和传输功率计算传输速率
        task_size_offload = task_size.clone()
        task_size_offload[off_actions==0] = 0  # 任务不卸载时传输大小为0
        offloading_penalty = torch.div(task_size_offload, trans_rate+TINY)      # 卸载时间
        # 总传输次数
        all_trans = off_actions[off_actions==1].bool().sum()
        server_comp_time = torch.div(task_size_offload*task_comp_eff, torch.gather(server_comp_frez, 1, server_selection))     # 服务器计算时间
        # 传输时间不大于T时
        # 每个任务的处理时间：本地排队时间+本地计算时间/卸载时间+服务器排队时间+服务器计算时间
            # 选择卸载则本地排队与计算时间为0
            # 选择本地计算则卸载时间与服务器计算时间为0
        process_time = local_comp_time + local_queue_time + torch.gather(self.server_queue_time, 1, server_selection) + server_comp_time      # 不考虑同一服务器所收到多个任务的处理先后顺序
        # 传输时间大于T的任务传输失败，给予惩罚
        offloading_penalty[offloading_penalty<=self.args.delta_t] = 0     # 传输成功不惩罚
        offloading_penalty[offloading_penalty>self.args.delta_t] = 500       # 传输失败进行惩罚
        # offloading_penalty[offloading_penalty>self.args.delta_t] = offloading_penalty[offloading_penalty>self.args.delta_t] * np.exp(fail_rate)**5
        # 传输失败次数
        trans_fail = offloading_penalty[offloading_penalty>0].bool().sum()
        rewards = process_time + offloading_penalty
        # 更新所有用户和服务器的状态
        graphs = self.state_update(local_comp_time, server_comp_time, server_selection, next_task)
        # reward = offloading_time.mean(-1)
        # 优化目标：任务整体计算时间最短、传输失败的概率最小
        return process_time, rewards, graphs, trans_fail, all_trans
    
    def transmission_rate(self, off_actions, server_selection):
        
        # 计算传输干扰
        noise = 1e-13
        power = self.graphs['user'].p_max.reshape(-1, self.user_num)*off_actions       # 对于不卸载的用户，传输功率为0
        power = power.unsqueeze(-1).repeat(1, 1, self.server_num)       # 扩展成（batch， user_num, server_num）
        tmp_power = torch.zeros((power.shape[0], self.user_num, self.server_num), device=self.args.device)
        trans_power = torch.scatter(tmp_power, -1, server_selection.unsqueeze(-1), power)       # 未选中的服务器传输功率为0
        # servers_batches = self.graphs['server'].batch
        
        # u2s_index_src = self.graphs['user', 'u2s', 'server'].edge_index[0]
        # u2s_index_dst = self.graphs['user', 'u2s', 'server'].edge_index[1]
        pw = self.pathloss * trans_power        # 信道与功率乘积
        server_receive_power = pw.sum(1)       # 服务器接收到的信号功率
        server_receive_power = server_receive_power.unsqueeze(1).repeat((1, pw.shape[1], 1))     # 服务器接收到的信号功率进行加和并扩展
        interference = server_receive_power - pw       # 单个信号接收到的干扰
        rate = torch.div(pw, interference+noise).sum(-1)   # 传输速率
        rate = self.args.BW * torch.log2(1 + rate)      # 传输速率
        return rate

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train_model(user_num, server_num, layouts_num, max_user_num=75, max_server_num=15, loaders=None, user_gen_probs=None, train_mode=True, inherit_model=None, evaluate=True, passstep=False):
    # 创建layouts
    if loaders:
        loader = loaders
        user_gen_probs = user_gen_probs
    else:
        user_nums = np.int32(np.ones(layouts_num)*user_num)
        server_nums = np.int32(np.ones(layouts_num)*server_num)
        loader, user_gen_probs = generate_layouts(user_nums, server_nums, args)
    if passstep:
        return None
    # 创建环境
    env = Env(loader, user_gen_probs, user_num, server_num, args)
    # 初始化RL模型
    if inherit_model:
        Model = inherit_model[0]
        fixed_model = inherit_model[1]
    else:
        Model = DTARL(num_layers=2, user_input_dim=8, server_input_dim=5, hidden_dim=args.hidden_dim).to(args.device)
        fixed_model = FixedRL(hidden_dim=args.hidden_dim, user_dim=8, server_dim=5, max_user_num=max_user_num, max_server_num=max_server_num).to(args.device)
    runner = Runner(env, Model, loader, args)
    if train_mode:
        if inherit_model:       # 继承模型 并微调
            model, fixed_model = runner.run_train(user_num, server_num, fixed_model=fixed_model, evaluate=True, finetune_lr=1e-5)
            return model, fixed_model, loaders, user_gen_probs
        else:   # 训练模型
            model, fixed_model = runner.run_train(user_num, server_num, fixed_model=fixed_model, evaluate=evaluate)
            return model, fixed_model, loaders, user_gen_probs
    else:   # 集成模型并训练
        runner.run_test(user_num, server_num, fixed_model)

def inherit_experiment(layouts_num):
    # print('-------------------------server num == 3-------------------------')
    # train_model(15, 3, layouts_num)
    # for server_num in range(10, 11):
    #     print('-------------------------server num == {}-------------------------'.format(server_num))
    #     Model = torch.load('./models/DTARL_{}_{}_{}.pt'.format(5*(server_num-1), server_num-1, args.n_steps), map_location=args.device)
    #     fixed_model = torch.load('./models/FixedRL_{}_{}_{}.pt'.format(5*(server_num-1), server_num-1, args.n_steps), map_location=args.device)
    #     train_model(5*server_num, server_num, layouts_num, train_mode=True, inherit_model=[Model, fixed_model])
    Model = torch.load('./models/DTARL_50_10_{}.pt'.format(args.n_steps), map_location=args.device)
    fixed_model = torch.load('./models/FixedRL_50_10_{}.pt'.format(args.n_steps), map_location=args.device)
    for server_num in range(11, 16):
        print('-------------------------inference server num == {}-------------------------'.format(server_num))
        # if server_num==14 or server_num == 15:
        train_model(5*server_num, server_num, layouts_num, train_mode=False, inherit_model=[Model, fixed_model])
        # else:
        #     train_model(5*server_num, server_num, layouts_num, passstep=True)

if __name__=='__main__':
    setup_seed(40)


    # 直接训练，对比fixed model和DTA-RL的rewards，process time，OFR
    train_model(15, 3, args.train_layouts, train_mode=True, evaluate=False)
    setup_seed(40)
    train_model(30, 6, args.train_layouts, train_mode=True, evaluate=False)
    setup_seed(40)
    train_model(45, 9, args.train_layouts, train_mode=True, evaluate=False)
    

    # Optimization Performance in Different Scales
    model, fixed_model, loaders, user_gen_probs = train_model(15, 3, args.train_layouts, max_user_num=15, max_server_num=3, train_mode=True, evaluate=False)
    train_model(15, 3, args.train_layouts, max_user_num=15, max_server_num=3, loaders=loaders, user_gen_probs=user_gen_probs, inherit_model=[model, fixed_model], train_mode=False, evaluate=True)
    setup_seed(40)
    model, fixed_model, loaders, user_gen_probs = train_model(30, 6, args.train_layouts, max_user_num=30, max_server_num=6, train_mode=True, evaluate=False)
    train_model(30, 6, args.test_layouts, max_user_num=30, max_server_num=6, loaders=loaders, user_gen_probs=user_gen_probs, inherit_model=[model, fixed_model], train_mode=False, evaluate=True)
    setup_seed(40)
    model, fixed_model, loaders, user_gen_probs = train_model(45, 9, args.train_layouts, max_user_num=45, max_server_num=9, train_mode=True, evaluate=False)
    train_model(45, 9, args.test_layouts, max_user_num=45, max_server_num=9, loaders=loaders, user_gen_probs=user_gen_probs, inherit_model=[model, fixed_model], train_mode=False, evaluate=True)
    

    # adaptbility performance
    inherit_experiment(args.train_layouts)

