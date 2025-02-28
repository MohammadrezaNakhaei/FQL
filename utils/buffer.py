import numpy as np 
import torch 


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_dim, action_dim, capacity, device, verbose=True):
        assert isinstance(obs_dim, int) and isinstance(action_dim, int)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        total_shape = 2*self.obs_dim + self.action_dim + 2
        total_bytes = 4*total_shape*capacity # 4 bytes for each tensor
        if verbose: print(f'Storage required: {total_bytes/1e9:.2f} GB') 
        storage_device = 'cpu' # by default
        if 'cuda' in device:
            mem_free, _ = torch.cuda.mem_get_info()
            # Heuristic: decide whether to use CUDA or CPU memory
            storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
        if verbose: print(f'Using {storage_device.upper()} memory for storage.')
        self.device = device
        self.storage_device = storage_device
        self.tensors = torch.empty((capacity, total_shape), dtype=torch.float32, device=storage_device)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done,):
        self.tensors[self.idx] = torch.as_tensor(
            np.concatenate([obs, action, [reward], next_obs, [done]]), 
            dtype=torch.float32)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        
    def add_torch(self, obs, action, reward, next_obs, done):
        for tensor in obs, action, reward, next_obs, done:
            assert tensor.ndim==2 
        batch_size = obs.shape[0]
        idxs = torch.arange(self.idx, self.idx+batch_size, device=self.storage_device)%self.capacity
        self.idx = (self.idx+batch_size)%self.capacity
        self.full = self.full or self.idx+batch_size>self.capacity
        self.tensors[idxs] = torch.cat([obs, action, reward, next_obs, done], dim=-1).to(self.storage_device) 

    def sample(self, batch_size, return_next_action=False):
        idxs = torch.randint(0, self.capacity if self.full else self.idx, size=(batch_size,), device=self.storage_device)
        tensor = self.tensors[idxs]
        if self.device=='cuda:0' and self.storage_device =='cpu':
            tensor = tensor.to('cuda:0')
        obss = tensor[:, :self.obs_dim]
        actions = tensor[:, self.obs_dim:self.obs_dim+self.action_dim]
        rewards = tensor[:, self.obs_dim+self.action_dim: self.obs_dim+self.action_dim+1]
        next_obss = tensor[:, -self.obs_dim-1:-1]
        dones = tensor[:, -1:]
        if return_next_action:
            next_actions = self.tensors[(idxs+1)%self.capacity, self.obs_dim:self.obs_dim+self.action_dim]
            return obss, actions, rewards, next_obss, dones, next_actions
        return obss, actions, rewards, next_obss, dones
    
    def save(self, path):
        torch.save(self.tensors[:len(self)].to('cpu'), path)

    def load(self, path):
        data = torch.load(path)
        assert data.shape[0]<=self.capacity, 'buffer is too small for the dataset'
        assert data.shape[1]==self.tensors.shape[1], 'buffer shape does not match the dataset'
        self.idx = data.shape[0]
        self.tensors[:self.idx] = data
        self.full = self.idx==self.capacity

    def load_d4rl(self, data):
        assert data['rewards'].shape[0]<=self.capacity, 'buffer is too small for the dataset'
        tensor = torch.as_tensor(
                    np.concatenate([data['observations'], data['actions'], data['rewards'][:, None], data['next_observations'], data['terminals'][:, None]], axis=1), 
                    dtype=torch.float32)
        self.tensors = tensor.to(self.storage_device) 
        self.idx = tensor.shape[0]