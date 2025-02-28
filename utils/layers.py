import copy
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state


def make_mlp(in_dim, mlp_dims, out_dim, act=None, layer_norm=False, dropout=0.):
    """
    MLP with Gelu activations, and optionally layer norm and dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(nn.Linear(dims[i], dims[i+1]))
        if layer_norm: mlp.append(nn.LayerNorm(dims[i+1]))
        if dropout and i==0:
            mlp.append(nn.Dropout(dropout))
        mlp.append(nn.GELU())
    mlp.append(nn.Linear(dims[-2], dims[-1]))
    if act:
        mlp.append(act())
    net = nn.Sequential(*mlp)
    net.apply(weight_init)
    return Mlp(net)



class Mlp(nn.Module):
	def __init__(self, net,):
		super().__init__()
		self.net = net 
	
	def forward(self, *args):
		return self.net(torch.cat(args, dim=-1))


def weight_init(m):
	"""Custom weight initialization for TD-MPC2."""
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)
	elif isinstance(m, nn.ParameterList):
		for i,p in enumerate(m):
			if p.dim() == 3: # Linear
				nn.init.trunc_normal_(p, std=0.02) # Weight
				nn.init.constant_(m[i+1], 0) # Bias
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			gain = nn.init.calculate_gain('relu')
			nn.init.orthogonal_(m.weight.data, gain)
			if hasattr(m.bias, 'data'):
				m.bias.data.fill_(0.0)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +(1 - tau) * target_param.data)


class Ensemble(nn.Module):
    """Vectorized ensemble of modules"""

    def __init__(self, modules, **kwargs):
        super().__init__()

        self.params_dict, self._buffers = stack_module_state(modules)
        self.params = nn.ParameterList([p for p in self.params_dict.values()])
        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        base_model = copy.deepcopy(modules[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))

        self.vmap = torch.vmap(
            fmodel, in_dims=(0, 0, None), randomness="different", **kwargs
        )
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap(self._get_params_dict(), self._buffers, torch.cat(args, dim=-1), **kwargs)

    def _get_params_dict(self):
        params_dict = {}
        for key, value in zip(self.params_dict.keys(), self.params):
            params_dict.update({key: value})
        return params_dict

    def __repr__(self):
        return "Vectorized " + self._repr