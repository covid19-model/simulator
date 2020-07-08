
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

    name = 'tracing-testing'
    start_date = '2021-01-01'
    end_date = '2021-03-02'
    random_repeats = 48
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 1

    # experiment parameters
    policies = [
        ('basic', 'fifo', 1, 30),
        ('basic', 'fifo', 3, 5000),
        ('advanced', 'exposure-risk', 1, 30),
    ]

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area
    cpu_count = args.cpu_count

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
        cpu_count=cpu_count,
        full_scale=full_scale,
        verbose=verbose,
    )

    # contact tracing experiment for various options
    for policy, queue, capacity_factor, contacts in policies:

        # no additional measures            
        m = []

        # set testing params via update function of standard testing parameters
        def test_update(d):
            d['test_smart_delta'] =  3 * TO_HOURS # 3 day time window considered for inspecting contacts
            d['test_smart_action'] = 'test' # test traced individuals
            d['test_targets'] = 'isym' 
            d['smart_tracing'] = policy
            d['test_smart_num_contacts'] = contacts
            d['test_queue_policy'] = queue
            d['tests_per_batch'] = capacity_factor * d['tests_per_batch'] # test capacity is artificially increased
            return d

        simulation_info = options_to_str(
            capacity_factor=capacity_factor,
            contacts=contacts, 
            policy=policy,
            queue=queue)
            
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

