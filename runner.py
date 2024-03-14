import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from collections import defaultdict
import pandas as pd

class Runner:
    def __init__(self, env, model, train_loader, args) -> None:
        self.buffer = []
        self.env = env
        self.model = model
        self.train_loader = train_loader
        self.args = args
        
    def run_train(self, user_num, server_num, fixed_model=None, evaluate=False, finetune_lr=None):
        rewards = []
        process_time_list = []
        policy_loss_list = []
        trans_fail_rate_list = []
        complete_local_process_time_list = []
        complete_off_process_time_list = []
        complete_off_trans_fail_rate_list = []
        complete_off_lessqueue_process_time_list = []
        complete_off_lessqueue_trans_fail_rate_list = []

        if finetune_lr:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=finetune_lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        lr_decay_step = self.args.n_steps//self.args.episode
        lr_decay_rate = 0.96
        self.opt_scheduler = lr_scheduler.MultiStepLR(self.optimizer, range(lr_decay_step), gamma=lr_decay_rate)
        
        if fixed_model:
            self.fixed_model = fixed_model
            if finetune_lr:
                self.fixed_optimizer = torch.optim.Adam(self.fixed_model.parameters(), lr=finetune_lr)      # 微调阶段优化器
            else:
                self.fixed_optimizer = torch.optim.Adam(self.fixed_model.parameters(), lr=self.args.learning_rate)
            lr_decay_step = self.args.n_steps//self.args.episode
            lr_decay_rate = 0.96
            self.fixed_opt_scheduler = lr_scheduler.MultiStepLR(self.fixed_optimizer, range(lr_decay_step), gamma=lr_decay_rate)

        graph = next(iter(self.train_loader))
        tasks = self.env.generate_tasks(self.args.episode)
        trans_fail_num_list = []
        all_trans_num_list  = []
        fixed_rewards_list = []
        fixed_process_time_list = []
        fixed_fail_rate_list = []
        for time_step in tqdm(range(self.args.n_steps)):
            
            if (time_step+1) % self.args.episode != 0:      # 没有到达一个episode，继续执行一个step
                one_step, graph, trans_fail_num, all_trans_num = self.generate_step(graph, tasks[time_step%self.args.episode])
                trans_fail_num_list.append(trans_fail_num.item())
                all_trans_num_list.append(all_trans_num.item())
                self.buffer.append(one_step)
                continue
            one_step, graph, trans_fail_num, all_trans_num = self.generate_step(graph, tasks[time_step%self.args.episode])
            trans_fail_num_list.append(trans_fail_num.item())
            all_trans_num_list.append(all_trans_num.item())
            # step数量达到一个episode，开始训练
            self.buffer.append(one_step)
            batch_episode = self.concat_batch(self.buffer)  # 把不同的step集合为一个batch
            # train模型
            reward = -batch_episode['r'].detach()       # 
            process_time = batch_episode['process_time']
            offloading_actions = batch_episode['off_a'].squeeze()     # 卸载动作
            offloading_log_probs = batch_episode['o_actions_lprob'].squeeze()     # 卸载的log概率
            s_lprobs = batch_episode['s_actions_lprob']     # 服务器选择概率
            
            off_policy_loss = -(offloading_log_probs.squeeze() * reward)
            selection_policy_loss = -(s_lprobs.squeeze() * reward)
            selection_policy_loss[offloading_actions.bool()] = 0
            if selection_policy_loss.shape[0] != 0:
                policy_loss = off_policy_loss.sum(0).mean() + selection_policy_loss.sum(0).mean()
            else:
                policy_loss = off_policy_loss.sum(0).mean()
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            self.opt_scheduler.step()
            if evaluate:
                # 全部选择本地计算
                self.env.restart_()
                graph = next(iter(self.train_loader))
                complete_local_process_time = self.complete_local_computing(graph, tasks)
                complete_local_process_time_list.append(complete_local_process_time.mean())
                # 选择卸载到最近的服务器
                self.env.restart_()
                graph = next(iter(self.train_loader))
                complete_off_process_time, complete_off_trans_fail_rate, complete_off_all_trans = self.complete_offload_computing(graph, tasks)
                complete_off_process_time_list.append(complete_off_process_time.mean())
                complete_off_trans_fail_rate_list.append(complete_off_trans_fail_rate.sum()/complete_off_all_trans.sum())
                # 选择队列最短的服务器
                self.env.restart_()
                graph = next(iter(self.train_loader))
                complete_off_lessqueue_process_time, complete_off_lessqueue_trans_fail_rate, complete_off_lessqueue_all_trans = self.complete_offload_computing_less_queue(graph, tasks)
                complete_off_lessqueue_process_time_list.append(complete_off_lessqueue_process_time.mean())
                complete_off_lessqueue_trans_fail_rate_list.append(complete_off_lessqueue_trans_fail_rate.sum()/complete_off_lessqueue_all_trans.sum())
            if fixed_model:
                fixed_process_time, fixed_rewards, fixed_trans_fail_num_list, fixed_all_trans_num_list= self.train_fixed(tasks)
                fixed_rewards_list.append(fixed_rewards.mean().item())
                fixed_process_time_list.append(np.mean(fixed_process_time))
                fixed_fail_rate_list.append(np.sum(fixed_trans_fail_num_list)/(np.sum(fixed_all_trans_num_list)+1e-20))
                # 输出fixed RL的信息
                print('fixed_rewards=={}, fixed_process time=={}, fixed_trans_fail_ratio=={}, fixed_trans_num=={}'.format(fixed_rewards.mean().item(), \
                                            np.mean(fixed_process_time), fixed_trans_fail_num_list.sum()/fixed_all_trans_num_list.sum(), fixed_all_trans_num_list.sum()))
            # 输出DTA-RL的训练信息
            print('policy_loss=={}, reward=={}, process time=={}, trans failed rate=={}, trans num=={}'.format(\
                    policy_loss.item(), -reward.mean().item(), process_time.mean().item(), np.sum(trans_fail_num_list)/np.sum(all_trans_num_list), np.sum(all_trans_num_list)))
            trans_fail_rate_list.append(np.sum(trans_fail_num_list)/np.sum(all_trans_num_list))
            process_time_list.append(process_time.mean().item())
            rewards.append(-reward.mean().item())
            trans_fail_num_list = []
            all_trans_num_list  = []
            
            self.buffer = []

            # 重新初始化用户和服务器的状态
            self.env.restart_()
            tasks = self.env.generate_tasks(self.args.episode)
            graph = next(iter(self.train_loader))

        if fixed_model and not evaluate:        # 只进行训练集收敛性对比
            compare_training_data = pd.DataFrame()  # DTA-RL和Fixed-RL对比训练数据
            compare_training_data['DTA_rewards'] = np.array(rewards)
            compare_training_data['Fixed_rewards'] = np.array(fixed_rewards_list)
            compare_training_data['DTA_process_time'] = np.array(process_time_list)
            compare_training_data['Fixed_process_time'] = np.array(fixed_process_time_list)
            compare_training_data['DTA_OFR'] = np.array(trans_fail_rate_list)
            compare_training_data['Fixed_OFR'] = np.array(fixed_fail_rate_list)

            compare_training_data.to_excel('./results/compare_training_data_{}_{}_{}.xlsx'.format(user_num, server_num, self.args.train_layouts))
        
        self.save_model(user_num, server_num, time_step)
        # plt.figure(figsize=(20, 8))
        # plt.subplot(1, 2, 1)
        # plt.plot(process_time_list)
        # plt.subplot(1, 2, 2)
        # plt.plot(policy_loss_list)
        # plt.savefig('./results/train_test_{}_{}.png'.format(user_num, server_num))

        train_data = pd.DataFrame()
        # train_data['rewards'] = np.array(rewards)
        train_data['process time'] = process_time_list
        train_data['trans_fail_rate'] = np.array(trans_fail_rate_list)
        if evaluate:
            train_data['local process time'] = np.array(complete_local_process_time_list)
            train_data['off process time'] = np.array(complete_off_process_time_list)
            train_data['off trans fail rate'] = np.array(complete_off_trans_fail_rate_list)
            train_data['off lessqueue process time'] = np.array(complete_off_lessqueue_process_time_list)
            train_data['off lessqueue trans fail rate'] = np.array(complete_off_lessqueue_trans_fail_rate_list)

            train_data['fixedRL process time'] = np.array(fixed_process_time_list)
            train_data['fixedRL trans fail rate'] = np.array(fixed_fail_rate_list)
        
        train_data.to_excel('./results/train_data_{}_{}.xlsx'.format(user_num, server_num))
        
        if fixed_model:
            return self.model, self.fixed_model
        else:
            return self.model


    def run_test(self, user_num, server_num, fixed_model):
        # 对预训练模型进行测试
        self.fixed_model = fixed_model
        graph = next(iter(self.train_loader))
        tasks = self.env.generate_tasks(self.args.episode)
        trans_fail_list = []
        all_trans_list  = []
        process_time_data = []
        trans_fail_data = []
        all_trans_data = []
        rewards_data = []

        fixedRL_process_time_data = []
        fixedRL_trans_fail_data = []
        fixedRL_all_trans_data = []
        
        complete_local_process_time_data = []
        complete_offload_process_time_data = []
        complete_offload_trans_fail_data = []
        complete_offload_all_trans_data = []


        complete_offload_lessqueue_process_time_data = []
        complete_offload_lessqueue_trans_fail_data = []
        complete_offload_lessqueue_all_trans_data = []


        for time_step in tqdm(range(self.args.n_steps)):
            # self.env.restart_(self.train_layouts, self.train_pathlosses)
            
            if (time_step+1) % self.args.episode != 0:      # 没有到达一个episode，继续执行一个step
                one_step, graph, trans_fail_num, all_trans_num = self.generate_step(graph, tasks[time_step%self.args.episode], train_mode=False)
                trans_fail_list.append(trans_fail_num.item())
                all_trans_list.append(all_trans_num.item())
                self.buffer.append(one_step)
                continue
            one_step, graph, trans_fail_num, all_trans_num = self.generate_step(graph, tasks[time_step%self.args.episode], train_mode=False)
            trans_fail_list.append(trans_fail_num.item())
            all_trans_list.append(all_trans_num.item())
            # step数量达到一个episode，开始训练
            self.buffer.append(one_step)
            batch_episode = self.concat_batch(self.buffer)  # 把不同的step集合为一个batch 
            process_time = batch_episode['process_time']
            reward = batch_episode['r']
            process_time_data.append(process_time.view(process_time.shape[0], -1).mean(-1).cpu().data.numpy())
            # rewards_data.append(reward.mean(-1).cpu().data.numpy())
            trans_fail_data.append(np.array(trans_fail_list))
            all_trans_data.append(np.array(all_trans_list))
            # 完全本地计算
            self.env.restart_()
            graph = next(iter(self.train_loader))
            complete_local_process_time_list = self.complete_local_computing(graph, tasks)
            complete_local_process_time_data.append(complete_local_process_time_list)
            # 选择最近的服务器
            self.env.restart_()
            graph = next(iter(self.train_loader))
            complete_off_process_time_list, complete_off_trans_fail_list, complete_off_all_trans_list = self.complete_offload_computing(graph, tasks)
            complete_offload_process_time_data.append(complete_off_process_time_list)
            complete_offload_trans_fail_data.append(complete_off_trans_fail_list)
            complete_offload_all_trans_data.append(complete_off_all_trans_list)
            #选择队列最短的服务器
            self.env.restart_()
            graph = next(iter(self.train_loader))
            complete_off_lessqueue_process_time, complete_off_lessqueue_trans_fail_rate, complete_off_lessqueue_all_trans = self.complete_offload_computing_less_queue(graph, tasks)
            complete_offload_lessqueue_process_time_data.append(complete_off_lessqueue_process_time)
            complete_offload_lessqueue_trans_fail_data.append(complete_off_lessqueue_trans_fail_rate)
            complete_offload_lessqueue_all_trans_data.append(complete_off_lessqueue_all_trans)

            # fixedRL测试
            fixed_process_time, fixed_rewards, fixed_trans_fail_num_list, fixed_all_trans_num_list= self.evaluate_fixed(tasks)
            fixedRL_process_time_data.append(np.array(fixed_process_time))
            fixedRL_trans_fail_data.append(np.array(fixed_trans_fail_num_list))
            fixedRL_all_trans_data.append(np.array(fixed_all_trans_num_list))
            # fixedRL_fail_rate_data.append(np.sum(fixed_trans_fail_num_list)/(np.sum(fixed_all_trans_num_list)+1e-20))


            print('process time=={}, trans failed rate=={}, trans num=={}'.format(-process_time.mean(), np.sum(trans_fail_list)/np.sum(all_trans_list), np.sum(all_trans_list)))
            trans_fail_list = []
            all_trans_list  = []
            self.buffer = []
            
            # 重新初始化用户和服务器的状态
            del batch_episode
            torch.cuda.empty_cache()
            self.env.restart_()
            tasks = self.env.generate_tasks(self.args.episode)
            graph = next(iter(self.train_loader))
        
        process_time_eachstep = np.stack(process_time_data).mean(0)
        # rewards_eachstep = np.stack(rewards_data).mean(0)
        trans_fail_eachstep = np.stack(trans_fail_data).mean(0)
        all_trans_eachstep = np.stack(all_trans_data).mean(0)
        
        process_time_eachepisode = np.stack(process_time_data).mean(1)
        # rewards_eachepisode = np.stack(rewards_data).mean(1)
        trans_fail_eachepisode = np.stack(trans_fail_data).mean(1)
        all_trans_eachepisode = np.stack(all_trans_data).mean(1)

        fixedRL_process_time_eachstep = np.stack(fixedRL_process_time_data).mean(0)
        fixedRL_trans_fail_eachstep = np.stack(fixedRL_trans_fail_data).mean(0)
        fixedRL_all_trans_eachstep = np.stack(fixedRL_all_trans_data).mean(0)

        fixedRL_process_time_eachepisode = np.stack(fixedRL_process_time_data).mean(1)
        fixedRL_trans_fail_eachepisode = np.stack(fixedRL_trans_fail_data).mean(1)
        fixedRL_all_trans_eachepisode = np.stack(fixedRL_all_trans_data).mean(1)

        # 每次采样取平均
        complete_local_process_time_eachstep = np.stack(complete_local_process_time_data).mean(0)
        complete_offload_process_time_eachstep = np.stack(complete_offload_process_time_data).mean(0)
        complete_offload_trans_fail_eachstep = np.stack(complete_offload_trans_fail_data).mean(0)
        complete_offload_all_trans_eachstep = np.stack(complete_offload_all_trans_data).mean(0)

        complete_local_process_time_eachepisode = np.stack(complete_local_process_time_data).mean(1)
        complete_offload_process_time_eachepisode = np.stack(complete_offload_process_time_data).mean(1)
        complete_offload_trans_fail_eachepisode = np.stack(complete_offload_trans_fail_data).mean(1)
        complete_offload_all_trans_eachepisode = np.stack(complete_offload_all_trans_data).mean(1)


        complete_offload_lessqueue_process_time_eachstep = np.stack(complete_offload_lessqueue_process_time_data).mean(0)
        complete_offload_lessqueue_trans_fail_eachstep = np.stack(complete_offload_lessqueue_trans_fail_data).mean(0)
        complete_offload_lessqueue_all_trans_eachstep = np.stack(complete_offload_lessqueue_all_trans_data).mean(0)


        complete_offload_lessqueue_process_time_eachepisode = np.stack(complete_offload_lessqueue_process_time_data).mean(1)
        complete_offload_lessqueue_trans_fail_eachepisode = np.stack(complete_offload_lessqueue_trans_fail_data).mean(1)
        complete_offload_lessqueue_all_trans_eachepisode = np.stack(complete_offload_lessqueue_all_trans_data).mean(1)


        
        test_data_each_step = pd.DataFrame()
        # test_data_each_step['rewards'] = rewards_eachstep
        test_data_each_step['process time'] = process_time_eachstep
        test_data_each_step['trans_fail_rate'] = trans_fail_eachstep/all_trans_eachstep
        test_data_each_step['fixedRL process time'] = fixedRL_process_time_eachstep
        test_data_each_step['fixedRL trans_fail_rate'] = fixedRL_trans_fail_eachstep/fixedRL_all_trans_eachstep
        test_data_each_step['local process time'] = complete_local_process_time_eachstep
        test_data_each_step['off process time'] = complete_offload_process_time_eachstep
        test_data_each_step['off trans fail rate'] = complete_offload_trans_fail_eachstep/complete_offload_all_trans_eachstep
        test_data_each_step['off lessqueue process time'] = complete_offload_lessqueue_process_time_eachstep
        test_data_each_step['off lessqueue trans fail rate'] = complete_offload_lessqueue_trans_fail_eachstep/complete_offload_lessqueue_all_trans_eachstep

        test_data_each_episode = pd.DataFrame()
        # test_data_each_episode['rewards'] = rewards_eachepisode
        test_data_each_episode['process time'] = process_time_eachepisode
        test_data_each_episode['trans_fail_rate'] = trans_fail_eachepisode/all_trans_eachepisode
        test_data_each_episode['fixedRL process time'] = fixedRL_process_time_eachepisode
        test_data_each_episode['fixedRL trans_fail_rate'] = fixedRL_trans_fail_eachepisode/fixedRL_all_trans_eachepisode
        test_data_each_episode['local process time'] = complete_local_process_time_eachepisode
        test_data_each_episode['off process time'] = complete_offload_process_time_eachepisode
        test_data_each_episode['off trans fail rate'] = complete_offload_trans_fail_eachepisode/complete_offload_all_trans_eachepisode
        test_data_each_episode['off lessqueue process time'] = complete_offload_lessqueue_process_time_eachepisode
        test_data_each_episode['off lessqueue trans fail rate'] = complete_offload_lessqueue_trans_fail_eachepisode/complete_offload_lessqueue_all_trans_eachepisode
        
        
        test_data_each_step.to_excel('./results/test_data_eachstep_{}_{}.xlsx'.format(user_num, server_num))
        test_data_each_episode.to_excel('./results/test_data_eachepisode_{}_{}.xlsx'.format(user_num, server_num))
    
    def train_fixed(self, tasks):
        self.env.restart_()
        graph = next(iter(self.train_loader))   # 初始化graph
        trans_fail_num_list = []
        all_trans_num_list = []
        process_time_list = []
        select_lprobs_list = []
        offloading_lprobs_list= []
        rewards = []
        offloading_actions_list = []
        for t in range(self.args.episode):
            process_time, reward, graph, trans_fail_num, all_trans_num, offloading_log_probs, select_log_probs,\
                  offloading_action, select_action = self.generate_step_fixedrl(graph, tasks[t])
            trans_fail_num_list.append(trans_fail_num.item())
            all_trans_num_list.append(all_trans_num.item())
            process_time_list.append(process_time.mean().item())
            offloading_actions_list.append(offloading_action)
            rewards.append(reward)
            offloading_lprobs_list.append(offloading_log_probs)
            select_lprobs_list.append(select_log_probs)
        offloading_lprobs_list = torch.vstack(offloading_lprobs_list)
        select_lprobs_list = torch.vstack(select_lprobs_list)
        rewards = torch.vstack(rewards)
        offloading_actions_list = torch.vstack(offloading_actions_list)
        
        off_policy_loss = (offloading_lprobs_list.squeeze() * rewards)
        selection_policy_loss = (select_lprobs_list.squeeze() * rewards)
        selection_policy_loss[offloading_actions_list.bool()] = 0
        if selection_policy_loss.shape[0] != 0:
            policy_loss = off_policy_loss.sum(0).mean() + selection_policy_loss.sum(0).mean()
        else:
            policy_loss = off_policy_loss.sum(0).mean()
        self.fixed_optimizer.zero_grad()
        policy_loss.backward()
        self.fixed_optimizer.step()
        self.fixed_opt_scheduler.step()
        return process_time_list, rewards, np.array(trans_fail_num_list), np.array(all_trans_num_list)

    def evaluate_fixed(self, tasks):
        self.env.restart_()
        graph = next(iter(self.train_loader))   # 初始化graph
        trans_fail_num_list = []
        all_trans_num_list = []
        process_time_list = []
        rewards = []
        for t in range(self.args.episode):
            process_time, reward, graph, trans_fail_num, all_trans_num, offloading_log_probs, select_log_probs,\
                  offloading_action, select_action = self.generate_step_fixedrl(graph, tasks[t], train_mode=False)
            trans_fail_num_list.append(trans_fail_num.item())
            all_trans_num_list.append(all_trans_num.item())
            process_time_list.append(process_time.mean().item())
            rewards.append(reward)
        rewards = torch.vstack(rewards)
        
        return process_time_list, rewards, np.array(trans_fail_num_list), np.array(all_trans_num_list)
    
    def generate_step(self, graph, next_task, train_mode=True):
        
        TINY = 1e-20
        offloading_actions_list = []
        select_actions_list = []
        select_log_probs_list = []

        offloading_log_probs_list = []
        select_log_probs_list = []

        offloading_Q, select_Q = self.model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)   # 每一个用户输出的Q, 全局embedding
        
        # select_Q转换回张量形式
        select_Q_tensor = torch.zeros((graph['user'].x.shape[0], graph['server'].x.shape[0]), device=self.args.device)  # [user_num * graph_num, server_num * graph_num]
        select_Q_tensor[graph['user', 'u2s', 'server'].edge_index[0], graph['user', 'u2s', 'server'].edge_index[1]] = select_Q

        user_batch = graph['user'].batch
        server_batch = graph['server'].batch
        # selections, one_hot_choice = self.sample_actions(u2s_source, u2s_target, select_Q)
        mask = user_batch.view(-1, 1)==server_batch
        # select_Q_tensor = select_Q_tensor[mask].view(-1, graph['server'].x.shape[0])
        select_Q_tensor = select_Q_tensor[mask].view(-1, self.env.user_num, self.env.server_num)
        offloading_Q_tensor = offloading_Q.view(-1, self.env.user_num, 2)
        if train_mode:  # 训练过程中采样
            select_sampler = torch.distributions.Categorical(select_Q_tensor)
            select_idx = select_sampler.sample()
            offloading_sampler = torch.distributions.Categorical(offloading_Q_tensor)
            offloading_idx = offloading_sampler.sample()         # now the idx has B elements
        else:
            offloading_idx = offloading_Q_tensor.argmax(dim=-1)
            select_idx = select_Q_tensor.argmax(dim=-1)

        select_probs = torch.gather(select_Q_tensor, -1, select_idx.unsqueeze(-1)).squeeze(-1)
        
        
        offloading_probs = torch.gather(offloading_Q_tensor, -1, offloading_idx.unsqueeze(-1)).squeeze(-1)

        offloading_actions_list.append(offloading_idx)
        select_actions_list.append(select_idx)

        # 被采样到的动作的log概率
        # offloading_log_probs = torch.log(offloading_Q[torch.arange(offloading_Q.shape[0]), offloading_idx]+TINY)
        offloading_log_probs = torch.log(offloading_probs+TINY)
        select_log_probs = torch.log(select_probs+TINY)

        offloading_log_probs_list.append(offloading_log_probs)
        select_log_probs_list.append(select_log_probs)
        

        offloading_actions_list = torch.vstack(offloading_actions_list)
        select_actions_list = torch.vstack(select_actions_list)

        offloading_log_probs_list = torch.vstack(offloading_log_probs_list)
        select_log_probs_list = torch.vstack(select_log_probs_list)


        process_time, reward, graphs, trans_fail_num, all_trans_num = self.env.selective_step(offloading_actions_list, select_actions_list, next_task)

        one_step = dict(off_a = offloading_actions_list.clone(),    # 是否卸载动作
                       s_a = select_actions_list.clone(),   # 服务器选择动作
                       r=reward.clone(),
                       process_time=process_time.clone(),
                       o_actions_lprob = offloading_log_probs_list.clone(),
                       s_actions_lprob = select_log_probs_list.clone()
                       )
        return one_step, graphs, trans_fail_num, all_trans_num

    def generate_step_fixedrl(self, graph, next_task, train_mode=True):
        user_batch = graph['user'].batch
        server_batch = graph['server'].batch
        offloading_Q, select_Q = self.fixed_model(user_batch, server_batch, graph.x_dict, graph.edge_index_dict)   # 每一个用户输出的Q, 全局embedding
        
        if train_mode:
            offloading_sampler = torch.distributions.Categorical(offloading_Q)
            offloading_idx = offloading_sampler.sample()         # now the idx has B elements
            select_sampler = torch.distributions.Categorical(select_Q)
            select_idx = select_sampler.sample()
        else:
            offloading_idx = offloading_Q.argmax(-1)
            select_idx = select_Q.argmax(-1)
        offloading_probs = torch.gather(offloading_Q, -1, offloading_idx.unsqueeze(-1)).squeeze(-1)
        offloading_lprob = torch.log(offloading_probs+1e-20)

        
        select_probs = torch.gather(select_Q, -1, select_idx.unsqueeze(-1)).squeeze(-1)
        select_lprob = torch.log(select_probs+1e-20)
        
        # u2s_source = graph['user', 'u2s', 'server'].edge_index[0]
        # u2s_target = graph['user', 'u2s', 'server'].edge_index[1]
        
        # one_hot_choice = torch.zeros_like(u2s_target, device=self.args.device).float()
        # one_hot_choice[select_actions[u2s_source]==u2s_target] = 1
        
        process_time, reward, graphs, trans_sucess_num, all_trans_num = self.env.selective_step(offloading_idx, select_idx, next_task)
        
        return process_time, reward, graphs, trans_sucess_num, all_trans_num, offloading_lprob, select_lprob, offloading_idx, select_idx
        


    def save_model(self, user_num, server_num, step):
        torch.save(self.model, './models/DTARL_{}_{}_{}.pt'.format(user_num, server_num, step+1))
        if self.fixed_model:
            torch.save(self.fixed_model, './models/FixedRL_{}_{}_{}.pt'.format(user_num, server_num, step+1))

    
    def sample_actions(self, source_index, target_index, probs):
        
        # Initialize the dictionary using defaultdict
        action_dict = defaultdict(list)

        # Fill the dictionary
        for source, target, prob in zip(source_index.cpu().data.numpy(), target_index.cpu().data.numpy(), probs):
            action_dict[source].append([target, prob])

        # Initialize the result
        result = {}
        one_hot_choice = []
        # Sample for each source
        for source, target_probs in action_dict.items():
            targets, probs = zip(*target_probs)
            probs = torch.stack(list(probs))
            targets = torch.tensor(targets,device=self.args.device)
            # probs = torch.tensor(probs)
            # probs /= probs.sum()  # Normalize the probabilities
            choice = torch.multinomial(probs, 1).item()     # 根据概率采样
            result[source] = torch.stack([targets[choice], probs[choice]])
            one_hot_source = torch.zeros_like(probs, device=self.args.device)
            one_hot_source[choice] = 1
            one_hot_choice.append(one_hot_source)
        one_hot_choice = torch.concat(one_hot_choice)
        
        return result, one_hot_choice


    def concat_batch(self, buffer):
        
        episode_buffer = dict(off_a = torch.cat([episode['off_a'].unsqueeze(0) for episode in buffer], dim=0),
                              s_a = torch.cat([episode['s_a'].unsqueeze(0) for episode in buffer], dim=0),
                              r = torch.cat([episode['r'].unsqueeze(0) for episode in buffer], dim=0),
                              process_time = torch.cat([episode['process_time'].unsqueeze(0) for episode in buffer], dim=0),
                              o_actions_lprob = torch.cat([episode['o_actions_lprob'].unsqueeze(0) for episode in buffer], dim=0),
                              s_actions_lprob = torch.cat([episode['s_actions_lprob'].unsqueeze(0) for episode in buffer], dim=0)
        )

        return episode_buffer

    
    def complete_local_computing(self, graph, tasks):
        process_time_list = []
        for next_task in tasks:
            process_time, graph = self.local_computing_step(graph, next_task)
            process_time_list.append(process_time)
        print('computing time in completely locally compute: {}'.format(np.mean(process_time_list)))
        return np.array(process_time_list)
    
    
    def local_computing_step(self, graph, next_task):
        # offloading_actions = torch.zeros((1, graph['user'].x.shape[0], 1), dtype=torch.long, device=self.args.device)
        offloading_actions = torch.zeros((graph['user'].batch.max()+1, self.env.user_num), dtype=torch.long, device=self.args.device)
        # select_actions = torch.zeros((1, graph['user'].x.shape[0]), dtype=torch.long, device=self.args.device)
        select_actions = torch.ones((graph['user'].batch.max()+1, self.env.user_num), dtype=torch.long, device=self.args.device)
        # one_hot_choice = graph['user', 'u2s', 'server'].edge_index[0].float()
        process_time, reward, graphs, _, _ = self.env.selective_step(offloading_actions, select_actions, next_task)
        return process_time.mean().item(), graphs

    def complete_offload_computing(self, graph, tasks):
        process_time_list = []
        trans_fail_list = []
        all_trans_list = []
        for next_task in tasks:
            process_time, reward, graph, trans_fail_num, all_trans_num = self.offloading_computing_step(graph, next_task)
            trans_fail_list.append(trans_fail_num.item())
            all_trans_list.append(all_trans_num.item())
            process_time_list.append(process_time)
        print('computing time in completely offloading compute: {}, trans_fail_rate: {}'.format(np.mean(process_time_list), np.sum(trans_fail_list)/np.sum(all_trans_list)))
        return np.array(process_time_list), np.array(trans_fail_list), np.array(all_trans_list)

    def offloading_computing_step(self, graph, next_task):
        # offloading_actions = torch.ones((1, graph['user'].x.shape[0], 1), dtype=torch.long, device=self.args.device)
        offloading_actions = torch.ones((graph['user'].batch.max()+1, self.env.user_num), dtype=torch.long, device=self.args.device)
        # 选择最近的服务器
        # select_actions, one_hot_choice = self.select_nearest_server(graph['user', 'u2s', 'server'].path_loss, graph['user', 'u2s', 'server'].edge_index[0],\
                                                    #  graph['user', 'u2s', 'server'].edge_index[1])
        select_actions = self.env.pathloss.argmax(-1)
        process_time, reward, graphs, trans_fail_num, all_trans_num = self.env.selective_step(offloading_actions, select_actions, next_task)
        return process_time.mean().item(), reward.mean().item(), graphs, trans_fail_num, all_trans_num

    def complete_offload_computing_less_queue(self, graph, tasks):
        process_time_list = []
        trans_fail_list = []
        all_trans_list = []
        for next_task in tasks:
            process_time, reward, graph, trans_fail_num, all_trans_num = self.offloading_computing_step_less_queue(graph, next_task)
            trans_fail_list.append(trans_fail_num.item())
            all_trans_list.append(all_trans_num.item())
            process_time_list.append(process_time)
        print('computing time completely offloading compute less queue: {}, trans_fail_rate: {}'.format(np.mean(process_time_list), np.sum(trans_fail_list)/np.sum(all_trans_list)))
        return np.array(process_time_list), np.array(trans_fail_list), np.array(all_trans_list)
    
    def offloading_computing_step_less_queue(self, graph, next_task):
        offloading_actions = torch.ones((graph['user'].batch.max()+1, self.env.user_num), dtype=torch.long, device=self.args.device)
        
        # 可到达的服务器队列长度
        server_queue = torch.ones((graph['user'].x.shape[0], graph['server'].x.shape[0]), device=self.args.device)
        server_queue[graph['user', 'u2s', 'server'].edge_index[0], graph['user', 'u2s', 'server'].edge_index[1]] = 0        # 有边的部分
        user_batch = graph['user'].batch
        server_batch = graph['server'].batch
        mask = user_batch.view(-1, 1)==server_batch
        server_queue = server_queue[mask].view(-1, self.env.user_num, self.env.server_num)
        server_queue = server_queue * 1e10      # 无法连接的服务器队列设置为无穷大
        server_queue = server_queue + self.env.server_queue_time.unsqueeze(1).repeat(1, self.env.user_num, 1)
        select_actions = server_queue.argmin(-1)
        process_time, reward, graphs, trans_fail_num, all_trans_num = self.env.selective_step(offloading_actions, select_actions, next_task)
        return process_time.mean().item(), reward.mean().item(), graphs, trans_fail_num, all_trans_num
        # process_time, reward, graphs
