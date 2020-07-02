
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

    name = 'conditional-measures'
    end_date = '2020-07-31'
    random_repeats = 96
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = None

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area

    # experiment parameters
    # Measures become active only if positive tests per week per 100k people exceed `max_pos_tests_per_week_per_100k`.
    # `intervention_times` can be set in order to allow changes in the policy only at discrete times,
    # if `None` measures can become active at any point.
    # `p_stay_home` determines how likely people are to follow the social distancing measures if they are active.
    # `beta_multiplier` determines scaling for `beta` when measure is active

    max_pos_tests_per_week_per_100k = 50
    is_measure_active_initially = False
    intervention_times = None
    calibrated_params = get_calibrated_params(country=country, area=area, multi_beta_calibration=False)
    p_stay_home = calibrated_params['p_stay_home']
    beta_multiplier = {'education': 0.0, 'social': 0.0, 'bus_stop': 1.0, 'office': 1.0, 'supermarket': 1.0}

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

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

    # conditional measures experiment
    max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    m = [
        UpperBoundCasesBetaMultiplier(t_window=Interval(0.0, TO_HOURS * max_days),
                                      beta_multiplier=beta_multiplier,
                                      max_pos_tests_per_week_per_100k=max_pos_tests_per_week_per_100k,
                                      intervention_times=intervention_times,
                                      init_active=is_measure_active_initially),

        UpperBoundCasesSocialDistancing(t_window=Interval(0, max_days * TO_HOURS),
                                        p_stay_home=p_stay_home,
                                        max_pos_tests_per_week_per_100k=max_pos_tests_per_week_per_100k,
                                        intervention_times=intervention_times,
                                        init_active=is_measure_active_initially)
        ]

    simulation_info = options_to_str(
        max_pos_tests_per_week_per_100k=50,
        initially_active=True,
        p_stay_home=p_stay_home
    )

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

