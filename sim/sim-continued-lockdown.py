
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
from lib.calibrationSettings import calibration_lockdown_dates, calibration_mob_paths, calibration_states, calibration_lockdown_beta_multipliers
from lib.calibrationFunctions import get_calibrated_params

TO_HOURS = 24.0

if __name__ == '__main__':

    name = 'continued-lockdown'
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
    # isolated_days = [7, 14] # how many days selected people have to stay in isolation
    # contacts_isolated = [10, 25] # how many contacts are isolated in the `test_smart_delta` window
    # policies = ['basic', 'advanced'] # contact tracing policies

    extended_lockdown_weeks = [2, 4, 8]
    calibrated_params = get_calibrated_params(country=country, area=area, multi_beta_calibration=False)
    p_stay_home = calibrated_params['p_stay_home']

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

    # Continue lockdown for different time periods
    for extended_lockdown_time in extended_lockdown_weeks:
        # measures
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        m = [
            SocialDistancingForAllMeasure(
                t_window=Interval(0.0, TO_HOURS * 7 * extended_lockdown_time),
                p_stay_home=p_stay_home),

            BetaMultiplierMeasureByType(
                t_window=Interval(0.0, TO_HOURS * 7 * extended_lockdown_time),
                beta_multiplier=calibration_lockdown_beta_multipliers)
            ]

        simulation_info = options_to_str(extended_lockdown_weeks=extended_lockdown_time,
                                         p_stay_home=p_stay_home)

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

