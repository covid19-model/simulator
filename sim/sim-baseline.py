
import sys
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
from lib.measures import *
from lib.experiment import Experiment, process_command_line
from lib.calibrationFunctions import get_calibrated_params

TO_HOURS = 24.0

if __name__ == '__main__':

    name = 'baseline'
    random_repeats = 48
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = None

    # set `True` for narrow-casting plot; should only be done with 1 random restart:
    store_mob = False 

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # command line parsing
    args = process_command_line()
    config = args.config
    cpu_count = args.cpu_count

    # Load calibrated parameters up to `maxBOiters` iterations of BO
    maxBOiters = 40 if config.area in ['BE', 'JU', 'RH'] else None
    calibrated_params = get_calibrated_params(config=config,
                                              multi_beta_calibration=False,
                                              maxiters=maxBOiters)

    # set simulation and intervention dates
    start_date = config.calibration_start_dates
    end_date = config.calibration_end_dates

    # create experiment object
    experiment_info = f'{name}-{config.country}-{config.area}'
    experiment = Experiment(
        experiment_info=experiment_info,
        start_date=start_date,
        end_date=end_date,
        random_repeats=random_repeats,
        cpu_count=cpu_count,
        full_scale=full_scale,
        verbose=verbose,
    )

    # baseline
    experiment.add(
        simulation_info='baseline',
        config=config,
        measure_list=[],
        lockdown_measures_active=False,
        seed_summary_path=seed_summary_path,
        set_calibrated_params_to=calibrated_params,
        set_initial_seeds_to=set_initial_seeds_to,
        full_scale=full_scale,
        store_mob=store_mob)

    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

