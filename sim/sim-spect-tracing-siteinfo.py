
import sys, os

from lib.mobilitysim import compute_mean_invariant_beta_multipliers
from lib.settings.beta_dispersion import get_invariant_beta_multiplier

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
from lib.calibrationFunctions import get_calibrated_params, get_calibrated_params_from_path
from lib.mobility_reduction import get_mobility_reduction

TO_HOURS = 24.0

if __name__ == '__main__':

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area
    cpu_count = args.cpu_count
    continued_run = args.continued

    name = 'spect-tracing-siteinfo'
    start_date = '2021-01-01'
    end_date = '2021-07-01'
    random_repeats = 100
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 5 / 7
    condensed_summary = True

    # ================ fixed contact tracing parameters ================
    smart_tracing_threshold = 0.0
    beacon_config = None
    area_population = 90546
    # ==================================================================

    # ============== variable contact tracing parameters ===============
    ps_adoption = [1.0, 0.5, 0.25, 0.1, 0.05, 0.0]
    isolation_caps = [0.005, 0.01, 0.02, 0.05, 0.1]
    manual_tracings = [dict(p_recall=0.0, p_manual_reachability=0.0, delta_manual_tracing=0.0)]
    # ==================================================================


    if args.p_adoption is not None:
        ps_adoption = [args.p_adoption]

    if args.beta_dispersion is not None:
        beta_dispersions = [args.beta_dispersion]

    if args.isolation_cap is not None:
        isolation_caps = [args.isolation_cap]

    if not args.calibration_state:
        calibrated_params = get_calibrated_params(country=country, area=area)
    else:
        calibrated_params = get_calibrated_params_from_path(args.calibration_state)
        print('Loaded non-standard calibration state.')

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # for debugging purposes
    if args.smoke_test:
        start_date = '2021-01-01'
        end_date = '2021-04-01'
        random_repeats = 2
        full_scale = False
        ps_adoption = [0.25]

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

    print('Using beta multipliers with invariance normalization.')
    beta_multipliers = {'education': 3.0,
                        'social': 6.0,
                        'bus_stop': 1/5.0,
                        'office': 4.0,
                        'supermarket': 2.0}
    beta_multipliers = compute_mean_invariant_beta_multipliers(beta_multipliers=beta_multipliers,
                                                               country=country, area=area,
                                                               max_time=28 * TO_HOURS,
                                                               full_scale=full_scale,
                                                               weighting='integrated_contact_time',
                                                               mode='rescale_all')
        # betas = {}
        # for key in beta_multipliers.keys():
        #     betas[key] = calibrated_params['beta_site'] * beta_multipliers[key]
        # calibrated_params['betas'] = betas
        # del calibrated_params['beta_site']

    # contact tracing experiment for various options
    for isolation_cap in isolation_caps:
        for p_adoption in ps_adoption:
            for k, manual_tracing in enumerate(manual_tracings):

                # measures
                max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

                m = [
                    # Beta multipliers
                    APrioriBetaMultiplierMeasureByType(beta_multiplier=beta_multipliers),

                    # Manual contact tracing
                    ManualTracingForAllMeasure(
                        t_window=Interval(0.0, TO_HOURS * max_days),
                        p_participate=1.0,
                        p_recall=manual_tracing['p_recall']),
                    ManualTracingReachabilityForAllMeasure(
                        t_window=Interval(0.0, TO_HOURS * max_days),
                        p_reachable=manual_tracing['p_manual_reachability']),

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

                # set testing params via update function of standard testing parameters
                def test_update(d):
                    d['smart_tracing_actions'] = ['isolate', 'test']
                    d['test_reporting_lag'] = 48.0
                    d['tests_per_batch'] = int(isolation_cap / 14 * area_population)
                    d['test_queue_policy'] = 'exposure-risk'

                    # isolation
                    d['smart_tracing_policy_isolate'] = 'advanced-global-budget'
                    d['smart_tracing_isolation_threshold'] = smart_tracing_threshold
                    d['smart_tracing_isolated_contacts'] = int(isolation_cap / 14 * area_population)
                    d['smart_tracing_isolation_duration'] = 14 * TO_HOURS,

                    # testing
                    d['smart_tracing_policy_test'] = 'advanced'
                    d['smart_tracing_testing_threshold'] = smart_tracing_threshold
                    d['smart_tracing_tested_contacts'] = 100000
                    d['trigger_tracing_after_posi_trace_test'] = False

                    # Tracing compliance
                    d['p_willing_to_share'] = 1.0
                    return d

                simulation_info = options_to_str(
                    p_adoption=p_adoption,
                    p_recall=manual_tracing['p_recall'],
                    p_manual_reachability=manual_tracing['p_manual_reachability'],
                    delta_manual_tracing=manual_tracing['delta_manual_tracing'],
                    isolation_cap=isolation_cap
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
                    expected_daily_base_expo_per100k=expected_daily_base_expo_per100k)

        print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

