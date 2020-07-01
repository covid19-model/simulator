
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

    name = 'k-groups'
    end_date = '2020-07-31'
    random_repeats = 96
    full_scale = True
    dry_run = False
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = None

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area

    # experiment parameters
    # Split citizens in `K_groups` groups and alternatingly install social distancing measures for the groups
    # `K_groups_weeks` determines for how many weeks this strategy is active

    K_groups = [2, 3, 4]
    K_groups_weeks = 6

    # seed
    c = 0
    np.random.seed(0)
    rd.seed(0)

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

    # Isolate k groups for different numbers of groups
    for groups in K_groups:
        # measures
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        m = [
            SocialDistancingForKGroups(
                t_window=Interval(0.0, (TO_HOURS * 7 * K_groups_weeks)),
                K=groups)
            ]

        simulation_info = options_to_str(
            K_groups=groups,
            K_groups_weeks=K_groups_weeks)

        experiment.add(
            simulation_info=simulation_info,
            country=country,
            area=area,
            measure_list=m,
            test_update=None,
            seed_summary_path=seed_summary_path,
            set_initial_seeds_to=set_initial_seeds_to,
            full_scale=full_scale)
    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

