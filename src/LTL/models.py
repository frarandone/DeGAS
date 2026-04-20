import matplotlib.pyplot as plt
import torch
from torch import distributions


def ou_model(time=50, init_x=-2., a=0.88, b=0.24, sigma=0.22):
    # a=0.88, b=0.24, \sigma=0.22. init 5 o -2
    # equilibrio a due, dinamica d raggiungimento equilibrio veloce
    # a=0.96, b=0.08, \sigma=0.18.
    # equilibrio a due ma dinamica piu lenta

    time = int(time)
    traj = torch.zeros(time)
    traj[0] = init_x
    noise = distributions.Normal(torch.tensor(0.), torch.tensor(sigma*sigma))

    for i in range(1 ,time):
        # update x
        traj[i] = a*traj[i-1] + b + noise.rsample()

    return traj



def logistic_model(time=50, init_x=1.0, r=0.18, k=10.0, sigma=0.1):

    #r=0.18,K=10,\sigma=0.10. init 1

    time = int(time)
    traj = torch.zeros(time)
    traj[0] = init_x
    noise = distributions.Normal(torch.tensor(0.), torch.tensor(sigma*sigma))

    for i in range(1, time):
        # update x
        traj[i] = traj[i - 1] + r * traj[i - 1] * (1 - traj[i - 1]/k)  + traj[i - 1] * noise.rsample()

    return traj



def double_well_model(time=50, init_x=-1.0, h = 0.15, a=1.0, b=1.0, sigma=0.65):

    #a=1, b=1, h=0.15, \sigma=0.65. init x=-1

    time = int(time)
    traj = torch.zeros(time)
    traj[0] = init_x
    noise = distributions.Normal(torch.tensor(0.), torch.tensor(sigma*sigma*h))

    for i in range(1, time):
        # update x
        traj[i] = traj[i - 1] + h * (a * traj[i - 1] - b * traj[i - 1] * traj[i - 1] * traj[i - 1])  +  noise.rsample()

    return traj


def simulate(model, n_traj, model_params):
    """
    Simulate a set of trajectories using the specified model and parameters.
    """
    traj_set = []
    for _ in range(n_traj):
        traj = model(**model_params)
        traj_set.append(traj)
    return torch.vstack(traj_set)


def plot_traj_set(traj_set, single_traj=0, color='blue', ls='-', label=None):
    """
    Plot a set of trajectories.
    """
    T = traj_set.shape[1]
    for i in range(single_traj):
        plt.plot(range(T), traj_set[i], color='grey')
    plt.plot(range(T), torch.mean(traj_set, 0), lw=3, ls=ls, color=color, label=label)
    plt.legend()