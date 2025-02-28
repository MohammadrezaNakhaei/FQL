from dataclasses import dataclass, field
import pyrallis
import sys, os

import random 
import numpy as np 
import torch
 
import d4rl, gym 
from gym.vector import SyncVectorEnv

import wandb, tqdm 
from utils.buffer import ReplayBuffer
from fql import FQL

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




def set_seed(seed):
	"""Set seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


@dataclass
class AgentConfig:
    hidden_dim: int = 512
    hidden_depth: int = 4
    num_q: int = 2
    lr: float = 3e-4
    flow_steps: int = 10
    discount: float = 0.99
    avg_target: bool = True
    normalize_q: bool = False
    tau: float = 0.005
    bc_weight: float = 10
    accelerate: bool = False

@dataclass
class TrainConfig:
    env_name: str = 'antmaze-umaze-v2'
    seed: int = 0
    device: str = 'cuda:0'
    epochs: int=100
    iter_per_epoch: int = 10000
    batch_size: int = 256
    num_evals: int=10
    fql: AgentConfig = field(default_factory=AgentConfig)
    use_wandb: bool = True
    project_name: str = 'FQL'
    
    
@pyrallis.wrap()
def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    with HiddenPrints():
        envs = SyncVectorEnv([lambda i=i: gym.make(cfg.env_name, seed=cfg.seed+i) for i in range(cfg.num_evals)])
    envs.action_space.seed(cfg.seed)
    env = envs.envs[0]
    
    cfg.fql.obs_dim = env.observation_space.shape[0]
    cfg.fql.act_dim = env.action_space.shape[0]
    cfg.fql.device = cfg.device

    agent = FQL(cfg.fql)
    print(agent)
    if cfg.use_wandb:
        logger = wandb.init(project=cfg.project_name, name=f'{cfg.env_name}-seed{cfg.seed}', config=cfg)
    data = d4rl.qlearning_dataset(env)
    if 'antmaze' in cfg.env_name:
        data['rewards'] -= 1
    buffer = ReplayBuffer(agent.obs_dim, agent.act_dim, data['rewards'].shape[0], device=cfg.device)
    buffer.load_d4rl(data)
    # vectorize evaluation
    def evaluate():
        total_reward = np.zeros(envs.num_envs)
        notdones = np.ones(envs.num_envs, dtype=np.bool_)
        obss = envs.reset()
        while np.any(notdones):
            actions = agent.select_action(obss, deterministic=True)
            next_obss, rewards, dones, _ = envs.step(actions)
            obss = next_obss
            total_reward[notdones] += rewards[notdones]
            for ind, done in enumerate(dones):
                if done:
                    obss[ind] = envs.envs[ind].reset()[0]
                    notdones[ind] = 0
        return total_reward.mean()
    
    for epoch in range(cfg.epochs):
        for itr in tqdm.trange(cfg.iter_per_epoch, desc=f'Epoch: {epoch+1}', miniters=25):
            batch = buffer.sample(cfg.batch_size)
            result = agent.update(batch)
            
        with HiddenPrints():
            reward = evaluate()
        d4rl_score = d4rl.get_normalized_score(cfg.env_name, reward)*100
        print(f'Normalized Reward: {d4rl_score:.2f}')
        if cfg.use_wandb:
            result = dict(result.to('cpu'))
            step = (epoch+1)*cfg.iter_per_epoch
            result.update({'step': step, 'eval/normalized_reward': d4rl_score, 'eval/step': step})
            logger.log(result, step=step)
            
    
if __name__ == '__main__':
    train()
    