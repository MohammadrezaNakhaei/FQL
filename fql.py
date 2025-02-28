from copy import deepcopy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.layers import make_mlp, Ensemble, soft_update_params
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule

torch.set_float32_matmul_precision('high')
class FQL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        self.obs_dim = cfg.obs_dim 
        self.act_dim = cfg.act_dim 
        self.device = cfg.device
        # flow policy: s,z,t->z_next policy: s,z->a
        self.flow_policy = make_mlp(self.obs_dim+self.act_dim+1, [cfg.hidden_dim]*cfg.hidden_depth, self.act_dim)
        self.policy = make_mlp(self.obs_dim+self.act_dim, [cfg.hidden_dim]*cfg.hidden_depth, self.act_dim)
        qs = [make_mlp(self.obs_dim+self.act_dim, [cfg.hidden_dim]*cfg.hidden_depth, 1, layer_norm=True) for _ in range(cfg.num_q)]
        self.qs = Ensemble(qs)
        self.target_qs = deepcopy(self.qs).requires_grad_(False)
        self.optim_flow = Adam(self.flow_policy.parameters(), lr=cfg.lr)
        self.optim_policy = Adam(self.policy.parameters(), lr=cfg.lr)
        self.optim_q = Adam(self.qs.parameters(), lr=cfg.lr)
        self.to(self.device)
        if cfg.accelerate:
            self._update = torch.compile(self._update)
            self._update = CudaGraphModule(self._update)
        
    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        assert obs.ndim==2
        num_env = obs.shape[0]
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x_0 = torch.zeros((num_env, self.act_dim), device=self.device) if deterministic else torch.randn((num_env, self.act_dim), device=self.device)
        action = self.policy(obs, x_0)
        return action.cpu().numpy()
    
    def update(self, batch):
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(batch)
    
    def _update(self, batch):
        obs, action, reward, next_obs, done = batch
        res = self._update_q(obs, action, reward, next_obs, done,)
        res.update(self._update_flow(obs, action))
        res.update(self._update_policy(obs, action))
        soft_update_params(self.qs, self.target_qs, self.cfg.tau)
        return res.mean()
    
    @torch.no_grad()
    def _get_flow_policy(self, obs, x_0, steps=10):
        batch_size = obs.shape[0]
        ones = torch.ones((batch_size, 1), device=self.device)
        for step in range(steps):
            t = step/steps*ones 
            v = self.flow_policy(obs, x_0, t)
            x_0 = x_0 + v/steps
        return x_0
    
    def _update_flow(self, obs, action, ):
        batch_size = obs.shape[0]
        # flow matching, sample initial x_0, t
        x_0 = torch.randn_like(action)
        t = torch.empty((batch_size, 1), device=self.device).uniform_(0, 1)
        x_t = t*action + (1-t)*x_0
        v = self.flow_policy(obs, x_t, t)
        self.optim_flow.zero_grad()
        loss = F.mse_loss(v, action-x_0)
        loss.backward()
        self.optim_flow.step()
        return TensorDict({'flow_loss':loss}).detach()
    
    def _update_q(self, obs, action, reward, next_obs, done, ):
        q = self.qs(obs, action)
        with torch.no_grad():
            x_0 = torch.randn_like(action)
            next_action = self.policy(next_obs, x_0)
            next_q = self.target_qs(next_obs, next_action)
            next_q = torch.mean(next_q, dim=0) if self.cfg.avg_target else torch.min(next_q, dim=0)[0]
            target_q = reward + self.cfg.discount*(1-done)*next_q
            target_q = target_q.broadcast_to(q.shape)
        critic_loss = F.mse_loss(q, target_q)
        self.optim_q.zero_grad()
        critic_loss.backward()
        self.optim_q.step()
        return TensorDict({'critic_loss':critic_loss, 'average_q': q.mean()}).detach()
    
    def _update_policy(self, obs, action):
        x_0 = torch.randn_like(action)
        flow_action = self._get_flow_policy(obs, x_0, steps=self.cfg.flow_steps)
        action = self.policy(obs, x_0)
        bc_loss = F.mse_loss(action, flow_action)
        q =self.qs(obs, action)
        lam = 1
        if self.cfg.normalize_q:
            with torch.no_grad():
                lam = 1 / q.abs().mean()
        policy_loss = -q.mean() * lam + bc_loss * self.cfg.bc_weight
        self.optim_policy.zero_grad()
        policy_loss.backward()
        self.optim_policy.step()
        return  TensorDict({'policy_loss':policy_loss, 'bc_loss': bc_loss}).detach()
    
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        repr = f'Flow Q-Learning\n'
        modules = ['Q-Functions', 'Policy', 'Flow Policy']
        for i, m in enumerate([self.qs, self.policy, self.flow_policy,]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,} M\n".format(self.total_params/1e6)
        return repr