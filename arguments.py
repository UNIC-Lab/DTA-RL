import argparse

args = argparse.ArgumentParser()
# 模型参数
args.add_argument('--learning_rate', default=1e-3)
args.add_argument('--input_dim', default=3)
args.add_argument('--hidden_dim', default=128)
args.add_argument('--alpha', default=0.02)
args.add_argument('--device', default='cuda:0')
args.add_argument('--two_hyper_layers', default=False)
args.add_argument('--hyper_hidden_dim', default=128)
args.add_argument('--num_hyper_layers', default=0)

# 实验环境参数
# args.add_argument('--user_num', default=15)
# args.add_argument('--server_num', default=3)
args.add_argument('--test_user_num', default=10)
args.add_argument('--p_min', default=0.6)
args.add_argument('--p_max', default=1)
args.add_argument('--task_min', default=1)
args.add_argument('--task_max', default=20,help='任务最大bit')
args.add_argument('--size_cof', default=1e6,help='任务大小系数')
args.add_argument('--task_comp_min', default=500,help='任务计算需求最小bit/cycle')
args.add_argument('--task_comp_max', default=2000,help='任务计算需求最大bit/cycle')
args.add_argument('--user_comp_min', default=50,help='用户最小计算能力cycle/s')
args.add_argument('--user_comp_max', default=100,help='用户最大计算能力cycle/s')
args.add_argument('--server_comp_min', default=500,help='服务器最小计算能力cycle/s')
args.add_argument('--server_comp_max', default=1000,help='服务器最大计算能力cycle/s')
args.add_argument('--comp_cof', default=1e7,help='计算能力系数')
args.add_argument('--train_layouts', default=64)
args.add_argument('--test_layouts', default=64)
args.add_argument('--len', default=300)
args.add_argument('--server_height', default=50)
args.add_argument('--carrier_f_start', default=2.4e9)
args.add_argument('--carrier_f_end', default=2.4835e9)
args.add_argument('--signal_cof', default=4.11)
args.add_argument('--noise_cof', default=-174, help='噪声功率')
args.add_argument('--Gr', default=17)
args.add_argument('--L', default=47.86)
args.add_argument('--BW', default=2e7, help='带宽')
args.add_argument('--delta_t', default=2, help='一帧时间长度')
args.add_argument('--lam', default=0.7, help='任务到达泊松分布参数')

args.add_argument('--n_steps', default=6000)
args.add_argument('--epochs', default=10)
args.add_argument('--evaluate_steps', default=10)
args.add_argument('--save_steps', default=50)
args.add_argument('--threshold', default=1000, help='超过1000m去除链接')
args.add_argument('--tau', default=0.01)
args.add_argument('--min_epsilon', default=0.001)
args.add_argument('--episode', default=60)
args.add_argument('--batch_size', default=4096)
args.add_argument('--train_batch', default=16)
args.add_argument('--offloading_penalty', default=30)

args = args.parse_args()
print(args)
