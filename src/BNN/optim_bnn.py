import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *
from time import time

from utils import get_data, mean_squared_error, mean_squared_error_bayes, neg_log_likelihood, neg_log_likelihood_one
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

torch.set_default_dtype(torch.float64)

args = [100, 2, 4]
N, D_X, D_H = args
X, Y, X_test = get_data(N=N, D_X=D_X)
X = X.unsqueeze(-1)
Y = Y.unsqueeze(-1)

bnn_one_pars = {'muw1': -1.0, 'muw2': 1.0, 'muw3': -1.0, 'muw4': -1.0, 'mub1': 1.0, 'mub2': -1.0, 'mub3': 1.0}

bnn_one_pars = {'muw1':1., 'muw2': -1., 'muw3': 1., 'muw4': -1., 'muw5': -1., 'muw6': 1., 'muw7': -1., 'muw8': -1.,  
            'mub1': -1., 'mub2': -1., 'mub3': -1., 'mub4': 1., 'mub5': 1.,
            'sigmaw1': 0.5, 'sigmaw2': 0.5, 'sigmaw3': 0.5, 'sigmaw4': 0.5, 'sigmaw5': 0.5, 'sigmaw6': 0.5, 'sigmaw7': 0.5, 'sigmaw8': 0.5,
            'sigmab1': 0.5, 'sigmab2': 0.5, 'sigmab3': 0.5, 'sigmab4': 0.5, 'sigmab5': 0.5,}


for key, value in bnn_one_pars.items():
    bnn_one_pars[key] = torch.tensor(value, requires_grad=True)    


compiledFile=compile2SOGA('../../programs/SOGA/Optimization/CaseStudies/bnn3.soga')
cfg = produce_cfg(compiledFile)
smooth_cfg(cfg)

lr = 0.001
steps = 500

#optimize(bnn_one_pars, neg_log_likelihood, Y.squeeze(-1).t(), cfg, steps=100, lr = 0.0001)

optimizer = torch.optim.Adam([bnn_one_pars[key] for key in bnn_one_pars.keys()], lr)

total_start = time()

batch_size = 20

for i in range(steps):

    optimizer.zero_grad()  # Reset gradients
    loss = 0
    for j in range(batch_size):
        sampled_index = np.random.randint(0, len(Y.squeeze([-1,1])))
        yj = Y.squeeze([1])[sampled_index].to(torch.float64)
        xj = X.squeeze([-1,1])[sampled_index]
        bnn_one_pars['x'] = xj.requires_grad_(False)

        current_dist = start_SOGA(cfg, bnn_one_pars, pruning='ranking')

        loss += neg_log_likelihood_one(yj, current_dist)

    # Backpropagate
    loss.backward(retain_graph=True)
    
    optimizer.step()

    # Print progress
    if i % int(steps/10) == 0:
        out = ''
        
        out = out + f" loss: {loss.item()}"

        #for key in bnn_one_pars.keys():
            #out = out + key + ': ' + str(bnn_one_pars[key].item()) + ' '
            
        print(out)

    total_end = time()

print('Optimization performed in ', round(total_end-total_start, 3))

y_means = []
for j in range(len(Y.squeeze([-1,1]))):
    yj = Y.squeeze([1])[j].to(torch.float64)
    xj = X.squeeze([-1,1])[j]
    bnn_one_pars['x'] = xj.requires_grad_(False)

    current_dist = start_SOGA(cfg, bnn_one_pars, pruning='ranking')
    y_means.append(current_dist.gm.mean()[current_dist.var_list.index('y')].detach().numpy())

# plot training data
plt.plot(X.squeeze(-1).numpy(), Y.squeeze(-1).numpy(), "kx")
# plot 90% confidence level of predictions
#plt.fill_between(X.numpy().flatten(), percentiles[0, :], percentiles[1, :], color="lightblue")
# plot mean prediction
plt.plot(X.numpy().flatten(), y_means, "blue", ls="solid", lw=2.0)
#ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")
plt.savefig('bnn3_optimization.png')

# save the parameters in a text file

with open('bnn_one_pars.json', 'w') as f:
    json.dump({key: value.item() for key, value in bnn_one_pars.items()}, f)