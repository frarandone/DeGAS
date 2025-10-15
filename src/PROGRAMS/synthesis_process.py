import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
print(sys.path)

from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from optimization import *
torch.set_default_dtype(torch.float64)
from PROGRAMS.data_generating_process import *
from PROGRAMS.likelihood import *
from PROGRAMS.programs import *
from PROGRAMS.run_pyro import *
import time

import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer import MCMC, NUTS
import pyro.distributions as dist
import pandas as pd
import arviz as az


def neg_log_likelihood(data, dist, idx):
    log_likelihood = torch.log(dist.gm.marg_pdf(data, idx))
    return - torch.sum(log_likelihood)

def run_optimization(name, data, params_init, true_params, eps, sensitivity_analysis=False, lr =0.05, steps = 100):
    soga_code = get_program(name)
    compiledFile = compile2SOGA_text(soga_code)
    cfg = produce_cfg_text(compiledFile)
    smooth_cfg(cfg, smooth_eps=eps)

    params_dict = initialize_params(params_init)  
    output_dist = start_SOGA(cfg, params_dict)
    data_var_list = get_vars(name)
    data_var_index = [output_dist.var_list.index(var) for var in data_var_list]
    data = torch.tensor(data, dtype=torch.float64)
    loss = lambda dist : neg_log_likelihood(data, dist, data_var_index)

    # repeat t
    loss_list, time, number_of_iterations = optimize(cfg, params_dict, loss, n_steps=steps, lr=lr, print_progress=False)

    if sensitivity_analysis:
        error_list = []
        #use the average error
        avg_error = np.mean([abs(params_dict[key].item() - true_value)/abs(true_value) for key, true_value in true_params.items()])
        return avg_error

    
        
    #for key, value in params_dict.items():
        #print(f"{key}: {value.item()}")

    # get the true value of the parameters from true_params and compute the relative error
    #for key, true_value in true_params.items():
       # estimated_value = params_dict[key].item()
       # error = abs(estimated_value - true_value)/abs(true_value)

    #calculate average error
    avg_error = np.mean([abs(params_dict[key].item() - true_value)/abs(true_value) for key, true_value in true_params.items()])
    # print(f"Average error: {avg_error}")

    return loss_list, params_dict, avg_error, time, number_of_iterations



#main function, taking as input the name of the program

if __name__ == "__main__":
    #take argument name from command line
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "bernoulli"
    print(f"Running optimization for program: {name}")

    program = name
    data_size = 1000
    true_params, init_params = get_params(program)
    data = generate_dataset(program, data_size, true_params)  
    print("DeGAS optimization")
    # run one optimization for lr in [0.01, 0.05, 0.1] and choose the best one
    best_lr = 0.01
    best_error = float('inf')
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]:
        try:
            loss_list, params, error, time_degas, number_of_iterations = run_optimization(program, data, init_params, true_params, eps=0.001, lr=lr, steps=500, sensitivity_analysis=False)
            if error < best_error:
                best_error = error
                best_lr = lr
        except Exception as e:
            print(f"Error occurred for lr={lr}: {e}")
            continue
    print(f"Best lr: {best_lr}")
    # now run the optimization with the best lr for 10 times and save in a file in losses_csv the average loss (with std) and in another file in errors_csv the R2 error
    losses = []
    parameters = []
    time_list = []
    iters_list = []
    for i in range(10):
        loss_list, params, error, time_degas, number_of_iterations = run_optimization(program, data, init_params, true_params, eps=0.001, lr=best_lr, steps=500)
        losses.append(loss_list)
        parameters.append(params)
        time_list.append(time_degas)
        iters_list.append(number_of_iterations)

    losses = np.array(losses)
    mean_loss = np.mean(losses, axis=0)
    std_loss = np.std(losses, axis=0)
    np.savetxt(f"losses_csv/losses_{program}.csv", np.vstack((mean_loss, std_loss)).T, delimiter=",", header="mean,std", comments='')

    errors = []
    for params in parameters:
        #calculate mean relative error
        error = np.mean([abs(params[key].item() - true_value)/abs(true_value) for key, true_value in true_params.items()])
        errors.append(error)
    errors = np.array(errors)

    mean_time = np.mean(time_list)
    print(f"Mean time: {mean_time}")
    np.savetxt(f"time_csv/time_{program}.csv", np.array([mean_time]), delimiter=",", header="time", comments='')

    mean_iters = np.mean(iters_list)
    #convert to int
    mean_iters = int(mean_iters)
    print(f"Mean iterations: {mean_iters}")
    np.savetxt(f"iters_csv/iters_{program}.csv", np.array([mean_iters]), delimiter=",", header="iterations", comments='')

    #save best learning rate
    np.savetxt(f"best_lr_csv/best_lr_{program}.csv", np.array([best_lr]), delimiter=",", header="best_lr", comments='')

    mean_error = np.mean(errors)
    print(f"Mean relative error: {mean_error}")
    np.savetxt(f"errors_csv/errors_{program}.csv", np.array([mean_error]), delimiter=",", header="R2", comments='')

    #save the params value in a csv file
    params_df = pd.DataFrame([{key: params[key].item() for key in params.keys()} for params in parameters])
    params_df.to_csv(f"opt_params/params_{program}.csv", index=False)


    # Sensitivity analysis
    print("Sensitivity analysis")
    error_list = []
    for eps in [0.1, 0.01, 0.001, 0.0001]:
        #print(f"Running optimization for eps={eps}")
        error = run_optimization(program, data, init_params, true_params, eps=eps, sensitivity_analysis=True, lr=best_lr, steps= mean_iters)
        error_list.append(error)

    # plot the error as a function of eps
    eps_values = [0.1, 0.01, 0.001, 0.0001]

    #print the error values
    for eps, error in zip(eps_values, error_list):
        print(f"Epsilon: {eps}, Error: {error}")

    ### PYRO INFERENCE

    print("Pyro inference")
    pyro.clear_param_store()
    model, guide = get_model_guide(program)
    #choose the best learning rate between [0.01, 0.05, 0.1]
    best_lr = 0.01
    best_error = float('inf')
    for lr in [0.01, 0.05, 0.1]:
        loss_list, iterations, time_VI = run_inference(model, guide, model_params=(data_size,torch.tensor(data, dtype=torch.float64)), n_steps=1000, lr=lr)
        #calculate mean relative error
        error = np.mean([abs(pyro.param(key + "_map").item() - true_value)/abs(true_value) for key, true_value in true_params.items()])
        if error < best_error:
            best_error = error
            best_lr = lr
    #with best lr run inference 10 times and save all the results
    print(f"Best lr for VI: {best_lr}")
    params_list = []
    loss_list_VI = []
    time_list_VI = []
    iters_list_VI = []
    for i in range(10):
        loss_list, iterations, time_VI = run_inference(model, guide, model_params=(data_size,torch.tensor(data, dtype=torch.float64)), n_steps=1000, lr=best_lr)
        params_list.append({key: pyro.param(key + "_map").item() for key in true_params.keys()})
        loss_list_VI.append(loss_list)
        time_list_VI.append(time_VI)
        iters_list_VI.append(iterations)
    errors = []
    for params in params_list:
        #calculate mean relative error
        error = np.mean([abs(params[key] - true_value)/abs(true_value) for key, true_value in true_params.items()])
        errors.append(error)
    errors = np.array(errors)
    mean_error = np.mean(errors)
    print(f"Mean iterations VI: {np.mean(iters_list_VI)}")
    print(f"Mean relative error VI: {mean_error}")
    np.savetxt(f"errors_csv/errors_VI_{program}.csv", np.array([mean_error]), delimiter=",", header="R2_VI", comments='')

    #for key, true_value in true_params.items():
            #estimated_value = pyro.param(key + "_map").item()
            #error = abs(estimated_value - true_value)/abs(true_value)
            #print(f"Error in {key}: {error}")

    #avg_error = np.mean([abs(pyro.param(key + "_map").item() - true_value)/abs(true_value) for key, true_value in true_params.items()])
    #print(f"Average error: {avg_error}")
    print(f"Mean time VI: {np.mean(time_list_VI)}")
    np.savetxt(f"time_csv/time_VI_{program}.csv", np.array([np.mean(time_list_VI)]), delimiter=",", header="time_VI", comments='')
    np.savetxt(f"iters_csv/iters_VI_{program}.csv", np.array([np.mean(iterations)]), delimiter=",", header="iterations_VI", comments='')
    #save the params value in a csv file
    params_df = pd.DataFrame(params_list)
    params_df.to_csv(f"opt_params/params_VI_{program}.csv", index=False)


    print("MCMC inference")
    rhat = 2.0
    num_samples = 0
    warmup_steps = 50

    while rhat > 1.05 and num_samples < 10000:
            num_samples += 500
            if num_samples > 2000:
                    warmup_steps += 50
            elif num_samples > 5000:
                    warmup_steps += 100

            # Run the inference with num_samples=500, warmup_steps=50 and check convergence with R-hat and Neff
            pyro.clear_param_store()
            # Do the same with MCMC
            nuts_kernel = NUTS(model, adapt_step_size=True)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
            start = time.time()
            mcmc.run((data_size, torch.tensor(data, dtype=torch.float64)))
            end = time.time()
            # Get total time
            total_time = end - start
            print(f"Total MCMC runtime: {total_time:.3f} seconds")
            #mcmc.summary()
            samples = mcmc.get_samples(group_by_chain=True)
            idata = az.convert_to_inference_data(samples)
            print(az.summary(idata, round_to=4))
            rhat_all = az.rhat(idata)
            valid_rhat = rhat_all.where(~np.isnan(rhat_all), drop=True)
            rhat = valid_rhat.to_array().max().item()
            neff = az.ess(idata).to_array().min().item()


    print(f"R-hat maximum: {rhat}")
    print(f"Neff minimum: {neff}")
    print(f"Num samples: {num_samples}")
    print(f"Warmup steps: {warmup_steps}")


    for key, true_value in true_params.items():
            estimated_value = torch.mean(samples[key])
            error = abs(estimated_value - true_value)/abs(true_value)
            print(f"Error in {key}: {error}")

    avg_error = np.mean([abs(torch.mean(samples[key]) - true_value)/abs(true_value) for key, true_value in true_params.items()])
    print(f"Average error: {avg_error}")

   
