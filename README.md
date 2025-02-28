# FQL
Unofficial Pytorch Implementation of [flow q-learning](https://arxiv.org/abs/2502.02538) 

## Overview
FQL is an offline RL algorithms utilizing flow matching (which is based on deterministic ode) to model behavior policy. 
The official [repository](https://github.com/seohongpark/fql) is based on Jax. This repo is for educational purposes and is limited to [D4RL](https://github.com/Farama-Foundation/D4RL) benchmark.

## Instructions
Install D4RL follwing this [repo](https://github.com/Farama-Foundation/D4RL):

Install dependencies from requirements.txt
```sh
pip install -r rquirement.txt
```

Train the agent:
``` sh
python train.py --env_name=antmaze-umaze-v2 --fql.bc_weight=10
```
By default, we are using **wandb** to log the results. You can turn it off by using ```--use_wandb=False```

It is possible to accelerate the training by using ```torch.compile``` and ```tensordict.nn.CudaGraphModule```. To do this set ```--fql.accelerate=True```. 

## Hyper-parameter Tunning
You can see the list of hyper-parameters in ``` train.py ``` in the corresponding``` dataclass```. The most imporant ones are:
* ```bc_weight```: This hyper-parameter is probably the most imporant one to tune. This balance maximizing the q function and remaining close to the bahavior policy (modeled with flow matching) when training the policy. The default value is 10.
* ``` normalize_q``` : If set to ``` True```, it will normalize the q values so that the ``` bc_weight``` will be invariant to the scale of q values similar to [*TD3BC*](https://github.com/sfujim/TD3_BC). In the paper, they did not normalize that and I followed them. It can useful when trying new environments. 
* ```avg_target```: In computing the target value for training the q networks, it will consider the average of the target networks instead of the minimum which is common in off-policy RL. If set to ```False```, then it will take the minimum (similar to clipped double Q-learning). 

## Coming soon
I will try to test FQL on [V-D4RL](https://github.com/conglu1997/v-d4rl) and compare the resutl to the baselines. 