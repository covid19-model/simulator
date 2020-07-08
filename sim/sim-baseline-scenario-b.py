
import sys, os
if '..' not in sys.path:
    sys.path.append('..')

import numpy as np
import random as rd
import pandas as pd
import pickle
import multiprocessing
import argparse
from lib.measures import *
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.calibrationSettings import calibration_lockdown_dates, calibration_mob_paths, calibration_states
from lib.calibrationFunctions import get_calibrated_params

TO_HOURS = 24.0

if __name__ == '__main__':

    name = 'baseline-scenario-b'
    start_date = '2021-01-01'
    end_date = '2021-03-02'
    random_repeats = 48
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 1

    # set `True` for narrow-casting plot; should only be done with 1 random restart:
    store_mob = False 

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area

    # Load calibrated parameters up to `maxBOiters` iterations of BO
    maxBOiters = 40 if area in ['BE', 'JU', 'RH'] else None
    calibrated_params = get_calibrated_params(country=country, area=area,
                                              multi_beta_calibration=False,
                                              maxiters=maxBOiters)

    # create experiment object
    experiment_info = f'{name}-{country}-{area}'
    experiment = Experiment(
        experiment_info=experiment_info,
        start_date=start_date,
        end_date=end_date,
        random_repeats=random_repeats,
        full_scale=full_scale,
        verbose=verbose,
    )

    # baseline
    experiment.add(
        simulation_info='baseline',
        country=country,
        area=area,
        measure_list=[],
        seed_summary_path=seed_summary_path,
        set_initial_seeds_to=set_initial_seeds_to,
        set_calibrated_params_to=calibrated_params,
        full_scale=full_scale,
        store_mob=store_mob,
        expected_daily_base_expo_per100k=expected_daily_base_expo_per100k)

    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

