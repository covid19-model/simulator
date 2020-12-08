
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
from lib.calibrationSettings import calibration_lockdown_dates, calibration_mob_paths, calibration_states, contact_tracing_adoption
from lib.calibrationFunctions import get_calibrated_params
from lib.settings.mobility_reduction import mobility_reduction

TO_HOURS = 24.0

if __name__ == '__main__':

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area
    cpu_count = args.cpu_count
    continued_run = args.continued

    name = 'beacon-spect-tracing'
    start_date = '2021-01-01'
    end_date = '2021-07-01'
    random_repeats = 100
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 5 / 7
    condensed_summary = True

    # contact tracing experiment parameters
    p_recall = 1.0
    p_manual_reachability = 0.1
    beacon_config = None

    if args.p_adoption is not None:
        ps_adoption = [args.p_adoption]
    else:
        ps_adoption = [1.0, 0.5, 0.25, 0.1, 0.05, 0.0]

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # Load calibrated parameters up to `maxBOiters` iterations of BO
    calibrated_params = get_calibrated_params(country=country, area=area,
                                              multi_beta_calibration=False,
                                              maxiters=None)
    # for debugging purposes
    if args.smoke_test:
        start_date = '2021-01-01'
        end_date = '2021-02-01'
        random_repeats = 2
        full_scale = False
        ps_adoption = [0.5]

    # create experiment object
    experiment_info = f'{name}-{country}-{area}'
    experiment = Experiment(
        experiment_info=experiment_info,
        start_date=start_date,
        end_date=end_date,
        random_repeats=random_repeats,
        cpu_count=cpu_count,
        full_scale=full_scale,
        condensed_summary=condensed_summary,
        continued_run=continued_run,
        verbose=verbose,
    )

    # contact tracing experiment for various options
    for p_adoption in ps_adoption:

        # measures
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        m = [
            # Manual contact tracing
            # infectors not compliant with digital tracing may reveal their mobility trace upon positive testing
            ManualTracingForAllMeasure(
                t_window=Interval(0.0, TO_HOURS * max_days),
                p_participate=1.0,
                p_recall=p_recall),
            # contact persons not compliant with digital tracing may be reached via phone
            ManualTracingReachabilityForAllMeasure(
                t_window=Interval(0.0, TO_HOURS * max_days),
                p_reachable=p_manual_reachability),
                
            # standard tracing measures
            ComplianceForAllMeasure(
                t_window=Interval(0.0, TO_HOURS * max_days),
                p_compliance=p_adoption),
            SocialDistancingForSmartTracing(
                t_window=Interval(0.0, TO_HOURS * max_days), 
                p_stay_home=1.0, 
                smart_tracing_isolation_duration=TO_HOURS * 14.0),
            SocialDistancingForSmartTracingHousehold(
                t_window=Interval(0.0, TO_HOURS * max_days),
                p_isolate=1.0,
                smart_tracing_isolation_duration=TO_HOURS * 14.0),
            ]
        
        if args.mobility_reduction:
            m += [
                # mobility reduction since the beginning of the pandemic
                SocialDistancingBySiteTypeForAllMeasure(
                    t_window=Interval(0.0, TO_HOURS * max_days),
                    p_stay_home_dict=mobility_reduction[country][area]),
            ]

        # set testing params via update function of standard testing parameters
        def test_update(d):
            d['smart_tracing_actions'] = ['isolate', 'test']
            d['test_reporting_lag'] = 48.0
            d['tests_per_batch'] = 100000

            # isolation
            d['smart_tracing_policy_isolate'] = 'advanced-threshold'
            d['smart_tracing_isolation_threshold'] =  0.1
            d['smart_tracing_isolated_contacts'] = 100000
            d['smart_tracing_isolation_duration'] = 14 * TO_HOURS,

            # testing
            d['smart_tracing_policy_test'] = 'advanced-threshold'
            d['smart_tracing_testing_threshold'] = 0.1
            d['smart_tracing_tested_contacts'] = 100000
            d['trigger_tracing_after_posi_trace_test'] = False
            
            return d

        simulation_info = options_to_str(
            p_adoption=p_adoption,
        )
            
        experiment.add(
            simulation_info=simulation_info,
            country=country,
            area=area,
            measure_list=m,
            beacon_config=None, 
            test_update=test_update,
            seed_summary_path=seed_summary_path,
            set_initial_seeds_to=set_initial_seeds_to,
            set_calibrated_params_to=calibrated_params,
            full_scale=full_scale,
            lockdown_measures_active=False,
            expected_daily_base_expo_per100k=expected_daily_base_expo_per100k)
                
    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()
