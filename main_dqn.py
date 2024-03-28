import os
import argparse

import numpy as np
import torch
torch.set_num_threads(4) 
import gym
from iabg.models.sim_env import BundleEnv
from iabg.models.model import StateEncoder,Actor, Critic
from iabg.collector import Collector
from iabg.trainer import offpolicy_trainer
from iabg.utils import set_random_seed
# from tianshou.data import Collector, VectorReplayBuffer
from tianshou.data import VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.policy import DQNPolicy ,ICMPolicy
# from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic
import setproctitle

from iabg.config import DATA_CONFIG, MODEL_CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-reward", type=float, default=100.0)
parser.add_argument("--stop-timesteps", type=int, default=1000000)
parser.add_argument('--dataset', type=str, default='clothing', choices=('movielens','clothing','electronic','food'))
parser.add_argument('--pool_size', type=int, default=50, help='pool size (default: 50)')
parser.add_argument('--bundle_size', type=int, default=10, help='bundle size (default: 3)')  #in movielens is maxsize
parser.add_argument('--env_mode', type=str, default='train', choices=('train', 'test'))
# parser.add_argument('--rew_mode', type=str, default='metric', choices=('item', 'compat', 'bundle', 'metric'))
parser.add_argument('--rew_mode', type=str, default='metric')
parser.add_argument('--metric', type=str, default='precision', choices=('precision', 'precision_plus', 'recall'))
parser.add_argument("--encoder", action="store_true")
parser.add_argument("--concat", action="store_true")
parser.add_argument("--embed-pretrain", action="store_true",default=False)
parser.add_argument("--fine-tune", action="store_true")
parser.add_argument("--gpu", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--n-step", type=int, default=3)
parser.add_argument("--target-update-freq", type=int, default=500)
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--net_pretrain_path", type=str, default=None)
parser.add_argument("--logfile", type=str, default="policy.pth")
parser.add_argument("--onetrm", default=False, action='store_true')
parser.add_argument("--whitening", default=False, action='store_true')
parser.add_argument("--pool_trm", default=False, action='store_true')
parser.add_argument("--nlp_dim", type=int, default=768)
parser.add_argument("--only_id", default=False, action='store_true')
parser.add_argument("--only_nlp", default=False, action='store_true')
parser.add_argument("--no_trm", default=False, action='store_true')
parser.add_argument("--query_trm", default=False, action='store_true')

if __name__ == "__main__":

    args = parser.parse_args()
    args.run = 'DQN'
    args.torch = True
    args.stop_timesteps = int(1e6)
    # args.pool_size = 100
    # args.bundle_size = 5
    args.rew_mode = ['item', 'metric']
    # args.rew_mode = ['metric']
    print(vars(args))


    env_config = {
        'dataset': args.dataset,
        'pool_size': args.pool_size,
        'bundle_size': args.bundle_size,
        'env_mode': args.env_mode,
        'rew_mode': args.rew_mode,
        'metric': args.metric,
        'item_model': 'NCF',
        'bundle_model': 'NCF',
        'compat_model': 'SG',
        'num_features': 1,
        'device': 'cpu',
        'onetrm': args.onetrm,
        'whitening': args.whitening,
        'pool_trm': args.pool_trm,
        'nlp_dim': args.nlp_dim,
        'only_id': args.only_id,
        'only_nlp': args.only_nlp,
        'no_trm': args.no_trm,
        'query_trm': args.query_trm,
    }
    env_config.update(DATA_CONFIG[env_config['dataset']])
    env_config['vocab_size'] = env_config['n_item']
    print(env_config)

    setproctitle.setproctitle('IABG@'+env_config['dataset'])

    custom_model_config = {
        'env_config': env_config,
        'model': 'iabg',
        'n_user': env_config['n_user'],
        'n_item': env_config['n_item'],
        'pool_size': env_config['pool_size'],
        'bundle_size': env_config['bundle_size'],
        'encoder': args.encoder,  # pool transformer encoder
        'concat': args.concat,  # concat bundle features
        'embed_path': None,  # use pretrained embedding
        'fine_tune': args.fine_tune,  # fine-tuning embedding
    }
    custom_model_config.update(MODEL_CONFIG[custom_model_config['model']])
    # custom_model_config['encoder'] = True
    # custom_model_config['concat'] = True
    # https://www.thinbug.com/q/54924582
    # https://blog.csdn.net/wen_fei/article/details/83117324
    # if args.embed_pretrain:
    #     custom_model_config['embed_path'] = '/root/reclib/iabg/output/models/movielens-SkipGramModel.npy'
    print(custom_model_config)
    set_random_seed(422)

    #Environment
    env = BundleEnv(env_config)
    test_env_config = env_config.copy()
    test_env_config['env_mode'] = 'test'
    train_envs = DummyVectorEnv([lambda: BundleEnv(env_config) for _ in range(2)])
    test_envs = DummyVectorEnv([lambda: BundleEnv(test_env_config) for _ in range(1)])
    # train_envs.seed([2,3])
    # test_envs.seed([6])

    #Policy
    device = torch.device("cuda:"+str(args.gpu) if args.gpu is not None and torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    pool_len = env.observation_space['pool'].shape[0]
    net = StateEncoder(env_config,[pool_len,4], pool_len, device=device, onetrm=args.onetrm).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=5e-5)

    if args.net_pretrain_path:
        net.load_state_dict(torch.load(args.net_pretrain_path, map_location=device), strict=False)
        print("Loaded net from: ", args.net_pretrain_path)

    policy = DQNPolicy(net, optim,
        args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join("save", args.dataset, args.logfile))

    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=device), strict=False)
        print("Loaded agent from: ", args.resume_path)


    #Collector
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(2000,len(train_envs)))
    test_collector = Collector(policy, test_envs, VectorReplayBuffer(2000,len(test_envs)))

    #Trainer
    result = offpolicy_trainer(policy,
        train_collector,test_collector,
        max_epoch=1000,
        step_per_epoch=4000,
        repeat_per_collect=1,
        episode_per_test=1000,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda mean_reward: mean_reward >= 195,     # Todo: 如果reward够高就直接stop，需要重新定义
        save_best_fn=save_best_fn)
    

    # if args.run == "DQN":
    #     cfg = {
    #         # TODO(ekl) we need to set these to prevent the masked values
    #         # from being further processed in DistributionalQModel, which
    #         # would mess up the masking. It is possible to support these if we
    #         # defined a custom DistributionalQModel that is aware of masking.
    #         "hiddens": [],
    #         "dueling": False,
    #         "exploration_config": {
    #             # The Exploration class to use.
    #             "type": "EpsilonGreedy",
    #             # Config for the Exploration class' constructor:
    #             "initial_epsilon": 1.0,
    #             "final_epsilon": 0.02,
    #             "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.

    #             # For soft_q, use:
    #             # "exploration_config" = {
    #             #   "type": "SoftQ"
    #             #   "temperature": [float, e.g. 1.0]
    #             # }
    #         },
    #         # Size of the replay buffer. Note that if async_updates is set, then
    #         # each worker will have a replay buffer of this size.
    #         "buffer_size": int(1e6)
    #     }
    # else:
    #     cfg = {}

 
