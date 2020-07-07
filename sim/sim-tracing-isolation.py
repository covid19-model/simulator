
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

    name = 'tracing-isolation'
    start_date = '2021-01-01'
    end_date = '2021-05-01'
    random_repeats = 96
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 1

    # experiment parameters
    isolate_days_list = [7, 14] # how many days selected people have to stay in isolation 
    contacts = 5000 # how many contacts are isolated in the `test_smart_delta` window at most
    policy = 'basic' # since all traced individuals are isolated, 'basic' == 'advanced'

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

    # contact tracing experiment for various options
    for isolate_days in isolate_days_list:

        # measures
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        
        m = [
            SocialDistancingForSmartTracing(
                t_window=Interval(0.0, TO_HOURS * max_days), 
                p_stay_home=1.0, 
                test_smart_duration=TO_HOURS * isolate_days),
            SocialDistancingForSmartTracingHousehold(
                t_window=Interval(0.0, TO_HOURS * max_days),
                p_isolate=1.0,
                test_smart_duration=TO_HOURS * isolate_days),
        ]

        # set testing params via update function of standard testing parameters
        def test_update(d):
            d['test_smart_delta'] =  3 * TO_HOURS # 3 day time window considered for inspecting contacts
            d['test_smart_action'] = 'isolate' # isolate traced individuals
            d['test_targets'] = 'isym' 
            d['smart_tracing'] = policy
            d['test_smart_num_contacts'] = contacts
            return d

        simulation_info = options_to_str(
            isolate_days=isolate_days, 
            contacts=contacts, 
            policy=policy)
            
        experiment.add(
            simulation_info=simulation_info,
            country=country,
            area=area,
            measure_list=m,
            test_update=test_update,
            seed_summary_path=seed_summary_path,
            set_initial_seeds_to=set_initial_seeds_to,
            set_calibrated_params_to=calibrated_params,
            full_scale=full_scale,
            expected_daily_base_expo_per100k=expected_daily_base_expo_per100k)
            
    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

