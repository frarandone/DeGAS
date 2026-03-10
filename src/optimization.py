import torch
from time import time

# SOGA modules
from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *

# The function optimize performs the optimization of the parameters using the Adam optimizer.
def optimize(cfg, params_dict, loss_func, n_steps=100, lr=0.05):

    # creates the optimizer, passing the parameters of the program as the parameters to optimize
    optimizer = torch.optim.Adam([params_dict[key] for key in params_dict.keys()], lr=lr)

    total_start = time()

    for i in range(n_steps):

        optimizer.zero_grad()  # Reset gradients
    
        # loss
        current_dist = start_SOGAGPU(cfg, params_dict)     # we compute the output distribution for the current values of the parameters using start_SOGA
        # print(cfg)
        # print(params_dict)
        # print(current_dist)
        loss = loss_func(current_dist)        # we compute the loss using the set of trajectories and the current output distribution

        # Backpropagate
        loss.backward()
    
        # Update parameters
        optimizer.step()

        # Print progress
        if i % int(n_steps/10) == 0:
            out = ''
            for key in params_dict.keys():
                out = out + key + ': ' + str(params_dict[key].item()) + ' '
            out = out + f" loss: {loss.item()}"
            # print(out)

    total_end = time()

    print(out)
    print('Optimization performed in ', round(total_end-total_start, 3))


def generate_trajectories(orig_model, n_traj, model_params=None, device='cuda:0'):
    traj_set = []
    for _ in range(n_traj):
        if model_params:
            # Pass device to model function
            traj = orig_model(**model_params, device=device)
        else:
            traj = orig_model(device=device)
        # Trajectories are already on correct device, just collect them
        traj_set.append(traj)
    traj_set = torch.vstack(traj_set)
    return traj_set

def initialize_params(pars, device='cuda:0'):
    params_dict = {}
    for key, value in pars.items():
        params_dict[key] = torch.tensor(value, requires_grad=True, device=device)
    return params_dict

# LOSSES 

# Loss: we use as loss the marginal negative log-likelihood of the trajectories given the output distribution.

def neg_log_likelihood(traj_set, dist, idx):
    log_likelihood = dist.gm.marg_log_pdf(traj_set[:, idx], idx)
    return - torch.sum(log_likelihood)



def L2_distance(traj_set, dist, idx):
    idx = [1,2,3,4,5,6,7,8,9]
    output_traj = dist.gm.mean()[idx]
    return torch.sum(torch.pow(traj_set[:, idx] - output_traj,2))



# used for PID
def signal_error(dist, target=3.14, eps=0.1, T=50):
    idx = list(range(1,T))
    # Get device from distribution instead of hardcoding .cuda()
    device = dist.get_device()
    if device >= 0:
        device = f'cuda:{device}'
    else:
        device = 'cpu'
    target_signal = target*torch.ones(len(idx), device=device)
    return torch.sum(torch.pow(dist.gm.mean()[idx] - target_signal, 2))