import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
print(sys.path)

from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *


from utils import generate_bnn_code_relu, generate_bnn_code_observe, generate_bnn_parameters, optimize, make_dataset, append_observe_block

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

torch.set_default_dtype(torch.float64)

args = [100, 2, 4]
N = 500
N_test = 50 
D_X = 2 
D_H = 4
n_hidden = 2
n_neurons = 3
#X, Y, X_test = get_data(N=N, D_X=D_X)

X, Y, X_test, Y_test = make_dataset(func_id=5, N=N, N_test=N_test, sigma=0.2)

X = X.unsqueeze(-1)
Y = Y.unsqueeze(-1)


soga_code = generate_bnn_code_relu(n=n_hidden, m=n_neurons)
#soga_code = generate_bnn_code_observe(n=n_hidden, m=n_neurons)
#soga_code = append_observe_block(soga_code, "x > 8", "y > 0")
constraint_str = """
x = _x;
if x > -0.3 {
    if x < 0.3 {
        observe(y > 2.5);
        observe(y < 3.);
    } else {
        skip;
    } end if;
} else {
    skip;
} end if;"""

soga_code = soga_code + "\n" + constraint_str
bnn_one_pars = generate_bnn_parameters(n=n_hidden, m=n_neurons)

compiledFile = compile2SOGA_text(soga_code)
cfg = produce_cfg_text(compiledFile)
smooth_cfg(cfg)

#remove from X the points where x is between -0.3 and 0.3
mask = (X.squeeze(-1).squeeze(-1) < -0.3) | (X.squeeze(-1).squeeze(-1) > 0.3)
X = X[mask]
Y = Y[mask]

print("start optimization")
bnn_one_pars = optimize(bnn_one_pars, cfg, X, Y, steps = 2000, lr = 0.01)


X_larger = torch.linspace(-4, 4, 100).unsqueeze(-1)

Y_larger = []
Y_larger_var = []
for j in range(len(X_larger)):
    xj = X_larger[j]
    bnn_one_pars['x'] = xj.requires_grad_(False)

    current_dist = start_SOGA(cfg, bnn_one_pars, pruning='ranking')
    Y_larger.append(current_dist.gm.mean()[current_dist.var_list.index('y')].detach().numpy())
    Y_larger_var.append(torch.sqrt(current_dist.gm.cov()[current_dist.var_list.index('y'), current_dist.var_list.index('y')]).detach().numpy())

#plot X_larger vs Y_larger with a confidence interval of 2 standard deviations
plt.figure(figsize=(10, 6))
plt.plot(X_larger.numpy().flatten(), Y_larger, "blue", ls="solid", lw=2.0, label='SOGA prediction')
plt.fill_between(X_larger.numpy().flatten(), 
                 (np.array(Y_larger) - 2*np.array(Y_larger_var)).flatten(), 
                 (np.array(Y_larger) + 2*np.array(Y_larger_var)).flatten(), 
                 color="lightblue", alpha=0.5, label='Confidence Interval (2 std)')
plt.scatter(X.numpy().flatten(), Y.numpy().flatten(), label="Training Data")
plt.savefig('bnn_C1.png')


df = pd.DataFrame({'X': X_larger.numpy().flatten(), 'Y_mean': Y_larger, 'Y_std': Y_larger_var})
df.to_csv('bnn_1_Cost1.csv', index=False)