# This is an example of how you can perform optimization using SOGA

from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *

from time import time
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# The original model simulates a thermostat system where the temperature is adjusted based on whether the thermostat is on or off.
# The two parameters `tOn` and `tOff` define the thresholds for turning the thermostat on and off.
# We assume we have some trajectories of temperature readings from this system, but we do not know the value of `tOn` and `tOff`.
# We use the function `orig_model` to generate synthetic temperature trajectories for the thermostat system 
# (setting the value of `tOn` and `tOff` to the true values of 18 and 20 respectively).


def orig_model(T=10, init_T=17., tOn=18, tOff=20, k=0.01, h=0.5, eps=0.1):
    traj = torch.zeros(T)
    isOn = False
    traj[0] = init_T
    noise = distributions.Normal(torch.tensor(0.), torch.tensor(eps))
        
    for i in range(1,T):
        
        #update temperature
        if isOn:
            traj[i] = traj[i-1] -k*traj[i-1] + h + noise.rsample() 
        else:
            traj[i] = traj[i-1] -k*traj[i-1] + noise.rsample() 

        # update thermostat state
        if isOn and traj[i] > tOff:
            isOn = False
        elif not isOn and traj[i] < tOn:
            isOn = True
            
    return traj     


# The function optimize performs the optimization of the parameters using the Adam optimizer.
def optimize(cfg, traj_set, params_dict, loss_func, n_steps=100):

    # creates the optimizer, passing the parameters of the program as the parameters to optimize
    optimizer = torch.optim.Adam([params_dict[key] for key in params_dict.keys()], lr=0.05)

    total_start = time()

    for i in range(n_steps):

        optimizer.zero_grad()  # Reset gradients
    
        # loss
        current_dist = start_SOGAGPU(cfg, params_dict)     # we compute the output distribution for the current values of the parameters using start_SOGA
        loss = loss_func(traj_set, current_dist)        # we compute the loss using the set of trajectories and the current output distribution

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
            print(out)

    total_end = time()

    print('Optimization performed in ', round(total_end-total_start, 3))


# Loss: we use as loss the marginal negative log-likelihood of the trajectories given the output distribution.

def neg_log_likelihood(traj_set, dist):

    # device = dist.get_device() # pytorch tensor get_device returns GPU id 0,1,etc or -1 (eg for CPU)
    # if device >= 0:
    #     device = f'cuda:{device}'
    # else:
    #     device = 'cpu'

    # print(device)
    # print(traj_set)
    idx = [1,2,3,4,5,6,7,8,9]
    log_likelihood = torch.log(dist.gm.marg_pdf(traj_set[:, idx], idx))
    # print(log_likelihood)
    return - torch.sum(log_likelihood)


if __name__ == '__main__':

    device = 'cuda:0'

    ## GENERATION OF TRAJECTORIES
	
    # We generate a set of trajectories using the original model
    # These trajectories will be used to compute the loss when fitting the parameters of the thermostat system.

    traj_set = []
    for _ in range(50):
        traj_set.append(orig_model())
    traj_set = torch.vstack(traj_set).to(device)

    ## SOGA Optimization

    # We open the SOGA model file `Thermostat.soga` that contains a program modeling the thermostat, in which `tOn` and `tOff` are declared as parameters.
    # We compile it and produce a smooth cfg from it.
    compiledFile=compile2SOGA('../programs/SOGA/Optimization/CaseStudies/Thermostat.soga')
    cfg = produce_cfg(compiledFile)
    smooth_cfg(cfg)

    # We set the initial values of the parameters `tOn` and `tOff` to 16 and 22 respectively (our guess, we will optimize this values based on the trajectories).
    pars = {'tOff':22., 'tOn':16.}
    params_dict = {}
    for key, value in pars.items():
        params_dict[key] = torch.tensor(value, requires_grad=True)    

    # We compute the output distribution for the initial value of the parameters (to plot later).
    initial_dist = start_SOGAGPU(cfg, params_dict)

    # We run the optimization loop
    optimize(cfg, traj_set, params_dict, neg_log_likelihood, n_steps=80)  # 80 is sufficient for the loss to converge

    # We compute the output distribution for the final value of the parameters
    final_dist = start_SOGAGPU(cfg, params_dict)

    import matplotlib.pyplot as plt

    ## PLOT 

    # We plot the first ten trajectories from original model and the mean trajectory
    for i in range(10):
        plt.plot(range(10), traj_set[i], color='grey')
    plt.plot(range(10), torch.mean(traj_set, 0), lw=3, color='blue', label='mean')

    # We plot the initial output distribution mean and the final output distribution mean
    plt.plot(range(10), initial_dist.gm.mean()[:10].detach(), lw=3, color='green', label='SOGA mean initial')
    plt.plot(range(10), final_dist.gm.mean()[:10].detach(), lw=3, color='orange', label='SOGA mean final')

    plt.legend()
    plt.show(block=True)


