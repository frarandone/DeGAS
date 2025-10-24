# DeGAS

## Usage

For all the experiments the hyperparameters are the ones specified in the paper, except for the synthesis for probabilistic programs that tunes automatically the hyperparamters when run on a single program.

Running the CPS experiments:

- Likelihood loss:
      folder src/CPS

      - All the experiments:  python3 cps_process.py 
      - Single experiment: python3 cps_process.py experiment_name

- Reachabilty loss:
      folder src/REACHABILITY

      - All the experiments:  python3 reachability_process.py 
      - Single experiment: python3 reachability_process.py experiment_name

with available experiments: thermostat, gearbox, bouncing_ball, pid


Running the Synthesis for Probabilistic Programs:


folder src/PROGRAMS

- All the experiments: python3 synthesis_process.py 
  Output -> results_all.csv structured as:
  VI Time, VI Error, MCMC Time, MCMC Error, MCMC r-hat, MCMC effective sample size, DeGAS Time, DeGAS Error

- One experiment: python3 synthesis_process.py experiment_name
  where experiment_name is one model of Table 1 of the paper written all lowercase


  