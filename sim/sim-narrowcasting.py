
import sys
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
from lib.measures import *
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.calibrationSettings import calibration_lockdown_dates, calibration_start_dates, calibration_lockdown_beta_multipliers
from lib.calibrationFunctions import get_calibrated_params, get_calibrated_params_from_path
from lib.mobility_reduction import get_mobility_reduction

TO_HOURS = 24.0

if __name__ == '__main__':

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area
    continued_run = args.continued

    name = 'narrowcasting'
    start_date = '2021-01-01'
    end_date = '2021-05-01'
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 5 / 7
    condensed_summary = False

    # set `True` for narrow-casting plot; should only be done with 1 random restart:
    store_mob = True
    cpu_count = 1
    random_repeats = 1

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    if not args.calibration_state:
        calibrated_params = get_calibrated_params(country=country, area=area)
    else:
        calibrated_params = get_calibrated_params_from_path(args.calibration_state)
        print('Loaded non-standard calibration state.')

    # experiment parameters
    mob_reduction = get_mobility_reduction('Germany', 'Baden-Württemberg', calibration_lockdown_dates['GER']['start'],
                                           calibration_lockdown_dates['GER']['end'])

    if args.smoke_test:
        start_date = '2021-01-01'
        end_date = '2021-02-15'
        full_scale = False

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

    # measures
    max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    m = [
        SocialDistancingBySiteTypeForAllMeasure(
            t_window=Interval(0.0, TO_HOURS * max_days),
            p_stay_home_dict=mob_reduction),
        ]

    m += [
        # standard tracing measures for non tracing experiments
        ComplianceForAllMeasure(
            t_window=Interval(0.0, TO_HOURS * max_days),
            p_compliance=0.0),
        SocialDistancingForSmartTracingHousehold(
            t_window=Interval(0.0, TO_HOURS * max_days),
            p_isolate=1.0,
            smart_tracing_isolation_duration=TO_HOURS * 14.0),
        ]

    # set testing params via update function of standard testing parameters
    def test_update(d):
        d['smart_tracing_actions'] = ['isolate', 'test']
        d['test_reporting_lag'] = 48.0

        # isolation
        d['smart_tracing_policy_isolate'] = 'basic'
        d['smart_tracing_isolated_contacts'] = 100000

        # testing
        d['smart_tracing_policy_test'] = 'basic'
        d['smart_tracing_tested_contacts'] = 100000
        d['trigger_tracing_after_posi_trace_test'] = False
        return d

    simulation_info = ''

    experiment.add(
        simulation_info=simulation_info,
        country=country,
        area=area,
        measure_list=m,
        test_update=test_update,
        seed_summary_path=seed_summary_path,
        set_calibrated_params_to=calibrated_params,
        set_initial_seeds_to=set_initial_seeds_to,
        full_scale=full_scale,
        expected_daily_base_expo_per100k=expected_daily_base_expo_per100k,
        store_mob=store_mob)

    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

