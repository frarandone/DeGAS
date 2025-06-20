# This is an example of how you can perform optimization using SOGA

from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *

from optimization import *

from time import time
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

from models import thermostat, gearbox, pid

thermo_params = {'model': thermostat, 
                 'n_traj': 50,
                 'script': '../programs/SOGA/Optimization/CaseStudies/Thermostat.soga',
                 'pars': {'tOff':22., 'tOn':16.},
                 'loss': lambda dist : neg_log_likelihood(traj_set, dist, idx=[1,2,3,4,5,6,7,8,9])
                 }
                 
gearbox_params = {'model': gearbox,
                  'n_traj': 50,
                  'script': '../programs/SOGA/Optimization/CaseStudies/Gearbox.soga',
                  'pars': {'s1':7., 's2':25.},
                  'loss': lambda dist : neg_log_likelihood(traj_set, dist, idx=list(range(1,20)))
                  }

pid_params = {'model': pid,
              'n_traj': 50,
              'script': '../programs/SOGA/Optimization/CaseStudies/PID.soga',
              'pars': {'s0':46., 's1':-23., 's2':0.},
              'loss': lambda dist : signal_error(dist)              
              }

model_list = {'thermostat': thermo_params,
              'gearbox': gearbox_params,
              'pid': pid_params,
              }

if __name__ == '__main__':

    for model_name, model in model_list.items():

        orig_model = model['model']
        if 'n_traj' in model.keys():
            n_traj = model['n_traj']
        else:
            n_traj = None
        script = model['script']
        pars = model['pars']
        loss = model['loss']

        print(f'Optimizing model {model_name}')

        ## GENERATION OF TRAJECTORIES
	
        # We generate a set of trajectories using the original model
        # These trajectories will be used to compute the loss when fitting the parameters of the thermostat system.

        if n_traj:
            traj_set = generate_trajectories(orig_model, n_traj)

        ## SOGA Optimization

        # We open the SOGA model file `Thermostat.soga` that contains a program modeling the thermostat, in which `tOn` and `tOff` are declared as parameters.
        # We compile it and produce a smooth cfg from it.
        compiledFile=compile2SOGA(script)
        cfg = produce_cfg(compiledFile)
        smooth_cfg(cfg)

        # We set the initial values of the parameters `tOn` and `tOff` to 16 and 22 respectively (our guess, we will optimize this values based on the trajectories).
        params_dict = initialize_params(pars)
    
        # We compute the output distribution for the initial value of the parameters (to plot later).
        initial_dist = start_SOGA(cfg, params_dict)

        # We run the optimization loop
        optimize(cfg, params_dict, loss, n_steps=80)  # 80 is sufficient for the loss to converge

        # We compute the output distribution for the final value of the parameters
        final_dist = start_SOGA(cfg, params_dict)

        print('\n\n')
