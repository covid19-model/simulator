
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

    name = 'baseline'
    end_date = '2020-07-31'
    random_repeats = 96
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
    country = args.country
    area = args.area

    # start simulation when lockdown ends
    start_date = calibration_lockdown_dates[country]['end']

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
        full_scale=full_scale,
        store_mob=store_mob)

    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

