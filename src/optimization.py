import torch
from time import time

# SOGA modules
from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *

# The function optimize performs the optimization of the parameters using the Adam optimizer.
def optimize(cfg, params_dict, loss_func, n_steps=100, lr=0.05, print_progress=True):

    # Standard deviations (params named "sigma...") are reparameterized as sigma = exp(raw)
    # and optimized in log-space. Two reasons: (1) it structurally guarantees sigma stays
    # strictly positive -- unconstrained Adam can otherwise walk it negative, which is
    # physically meaningless; (2) the likelihood of a Gaussian-mixture component is unbounded
    # as sigma -> 0+ (a real MLE singularity, not a bug -- see PROGRAMS/likelihood.py's module
    # docstring), so unconstrained optimization can chase arbitrarily tiny sigma to inflate
    # likelihood on a specific data point rather than genuinely fitting the distribution. In
    # log-space, d(loss)/d(raw) = d(loss)/d(sigma) * sigma, so the effective gradient shrinks
    # as sigma shrinks -- a self-damping effect that makes collapsing to a near-degenerate
    # sigma within a fixed step budget much harder, without capping how small a genuinely
    # well-supported sigma can end up. A sigma that starts EXACTLY at 0.0 is left untouched
    # (log(0) is undefined) -- that's DeGAS's own deliberate discrete/point-mass convention
    # (see sogaPreprocessor.py's compileBernoulli), which already receives no gradient at all
    # via PROGRAMS/likelihood.py's delta-branch handling, so there is nothing to reparameterize.
    raw_sigma = {}
    for key, value in params_dict.items():
        if key.startswith("sigma") and value.item() > 0:
            raw_sigma[key] = torch.log(value.detach().clone()).requires_grad_(True)

    def sync_reparameterized_sigmas():
        for key, raw in raw_sigma.items():
            params_dict[key] = torch.exp(raw)

    sync_reparameterized_sigmas()

    # creates the optimizer, passing the parameters of the program as the parameters to optimize
    trainable = [raw_sigma[key] if key in raw_sigma else params_dict[key] for key in params_dict.keys()]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    total_start = time()
    loss_list = []
    number_of_iterations = n_steps
    # Adam isn't guaranteed to end on its best iterate (it can overshoot/diverge in later
    # steps), so we track the best (params, loss) seen and snap back to it before returning --
    # otherwise callers reading params_dict alongside loss_list[-1] as "the optimized result"
    # can end up with a worse point than one Adam actually visited during this call.
    best_loss = None
    best_params = None
    for i in range(n_steps):

        optimizer.zero_grad()  # Reset gradients

        # loss
        current_dist = start_SOGA(cfg, params_dict)     # we compute the output distribution for the current values of the parameters using start_SOGA
        loss = loss_func(current_dist)        # we compute the loss using the set of trajectories and the current output distribution
        loss_list.append(loss.item())

        if best_loss is None or loss_list[-1] < best_loss:
            best_loss = loss_list[-1]
            best_params = {key: value.detach().clone() for key, value in params_dict.items()}

        # check for convergence within a tolerance of 1e-8 and wit a patience of 30 iterations
        if i > 30 and abs(loss_list[-1] - loss_list[-2]) < 1e-8 and all(abs(loss_list[-j] - loss_list[-j-1]) < 1e-8 for j in range(2, 31)):
            if print_progress:
                print(f"Converged at iteration {i}")
            number_of_iterations = i
            break

        # Backpropagate
        loss.backward(retain_graph=True)

        # Update parameters
        optimizer.step()
        sync_reparameterized_sigmas()  # rebuild params_dict's sigma entries as exp(raw) from
        # the just-stepped raw leaves, so the next iteration's start_SOGA call (and, if this
        # was the last iteration, the caller) sees the updated positive sigma values

        # Print progress
        if print_progress: #i % int(n_steps/10) == 0:
            out = ''
            for key in params_dict.keys():
                out = out + key + ': ' + str(params_dict[key].item()) + ' '
            out = out + f" loss: {loss.item()}"
            print(out)

    total_end = time()
    if print_progress:
        print('Optimization performed in ', round(total_end-total_start, 3))

    # Snap params_dict (and the trailing loss_list entry) back to the best iterate seen,
    # so a caller's "final" loss and "final" params are always mutually consistent. Reassigns
    # dict entries (rather than in-place .copy_()) since reparameterized sigma entries are
    # derived (non-leaf) tensors, not the leaves Adam actually optimized.
    if best_params is not None and best_loss < loss_list[-1]:
        with torch.no_grad():
            for key in params_dict.keys():
                params_dict[key] = best_params[key]
        loss_list[-1] = best_loss

    #put current dist mean and cov in a file
    #with open("current_dist_stats.txt", "a") as f:
        #f.write(f"Current dist mean: {current_dist.gm.mean().detach().numpy()}\n")
        #f.write(f"Current dist cov: {current_dist.gm.cov().detach().numpy()}\n")
    return loss_list, round(total_end-total_start, 3), number_of_iterations


def generate_trajectories(orig_model, n_traj, model_params=None):
    traj_set = []
    for _ in range(n_traj):
        if model_params:
            traj_set.append(orig_model(**model_params))
        else:
            traj_set.append(orig_model())
    traj_set = torch.vstack(traj_set)
    return traj_set

def initialize_params(pars):
    params_dict = {}
    for key, value in pars.items():
        params_dict[key] = torch.tensor(value, requires_grad=True)
    return params_dict

# LOSSES 

# Loss: we use as loss the marginal negative log-likelihood of the trajectories given the output distribution.

def neg_log_likelihood(traj_set, dist, idx):
    log_likelihood = torch.log(dist.gm.marg_pdf(traj_set[:, idx], idx))
    return - torch.sum(log_likelihood)



def L2_distance(traj_set, dist, idx):
    idx = [1,2,3,4,5,6,7,8,9]
    output_traj = dist.gm.mean()[idx]
    return torch.sum(torch.pow(traj_set[:, idx] - output_traj,2))



# used for PID
def signal_error(dist, target=3.14, eps=0.1, T=50):
    idx = list(range(1,T))
    target_signal = target*torch.ones(len(idx))
    return torch.sum(torch.pow(dist.gm.mean()[idx] - target_signal, 2))