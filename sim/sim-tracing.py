
import sys
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
from lib.measures import *
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.calibrationFunctions import get_calibrated_params, get_calibrated_params_from_path
from lib.distributions import CovidDistributions
from lib.calibrationSettings import calibration_lockdown_beta_multipliers


TO_HOURS = 24.0

if __name__ == '__main__':

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area
    cpu_count = args.cpu_count
    continued_run = args.continued

    name = 'tracing'
    start_date = '2021-01-01'
    end_date = '2021-05-01'
    random_repeats = 100
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 5 / 7
    condensed_summary = True

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    if not args.calibration_state:
        calibrated_params = get_calibrated_params(country=country, area=area)
    else:
        calibrated_params = get_calibrated_params_from_path(args.calibration_state)
        print('Loaded non-standard calibration state.')

    # contact tracing experiment parameters
    min_contact_time = 0.25  # hours

    if args.tracing_threshold is not None:
        smart_tracing_thresholds = [args.tracing_threshold]
    else:
        distr = CovidDistributions(country=country)
        smart_tracing_thresholds = [(min_contact_time * calibrated_params['beta_site']
                                    * (1 - np.exp(distr.gamma * (- distr.delta))))]

    if args.test_lag is not None:
        test_lags = [args.test_lag]
    else:
        test_lags = [48.0, 24.0, 3.0, 1.0]

    if args.p_adoption is not None:
        ps_adoption = [args.p_adoption]
    else:
        ps_adoption = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.0]
        # ps_adoption = [1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]


    if args.smoke_test:
        start_date = '2021-01-01'
        end_date = '2021-02-15'
        random_repeats = 1
        full_scale = False
        ps_adoption = [1.0]
        test_lags = [48.0]


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
    for smart_tracing_threshold in smart_tracing_thresholds:
        for test_lag in test_lags:
            for p_adoption in ps_adoption:

                # measures
                max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

                m = [
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
                    d['test_reporting_lag'] = test_lag

                    # isolation
                    d['smart_tracing_policy_isolate'] = 'advanced-threshold'
                    d['smart_tracing_isolation_threshold'] = smart_tracing_threshold
                    d['smart_tracing_isolated_contacts'] = 100000
                    d['smart_tracing_isolation_duration'] = 14 * TO_HOURS,

                    # testing
                    d['smart_tracing_policy_test'] = 'advanced-threshold'
                    d['smart_tracing_testing_threshold'] = smart_tracing_threshold
                    d['smart_tracing_tested_contacts'] = 100000
                    d['trigger_tracing_after_posi_trace_test'] = False

                    return d

                simulation_info = options_to_str(
                    p_adoption=p_adoption,
                    test_lag=test_lag,
                    tracing_threshold=smart_tracing_threshold,
                )

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

