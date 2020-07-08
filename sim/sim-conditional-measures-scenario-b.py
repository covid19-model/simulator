
import sys
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
from lib.measures import *
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.calibrationSettings import calibration_lockdown_dates, calibration_start_dates
from lib.calibrationFunctions import get_calibrated_params

TO_HOURS = 24.0

if __name__ == '__main__':

    name = 'conditional-measures-scenario-b'
    start_date = '2021-01-01'
    end_date = '2021-03-02'
    random_repeats = 48
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 1

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area

    # Load calibrated parameters up to `maxBOiters` iterations of BO
    maxBOiters = 40 if area in ['BE', 'JU', 'RH'] else None
    calibrated_params = get_calibrated_params(country=country, area=area,
                                              multi_beta_calibration=False,
                                              maxiters=maxBOiters)

    # experiment parameters
    # Measures become active only if positive tests per week per 100k people exceed `max_pos_tests_per_week_per_100k`.
    # `intervention_times` can be set in order to allow changes in the policy only at discrete times,
    # if `None` measures can become active at any point.
    # `p_stay_home` determines how likely people are to follow the social distancing measures if they are active.
    # `beta_multiplier` determines scaling for `beta` when measure is active

    max_pos_tests_per_week_per_100k = 50
    is_measure_active_initially = False
    intervention_times = None
    p_stay_home = calibrated_params['p_stay_home']
    beta_multiplier = calibration_lockdown_beta_multipliers

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # set simulation and intervention dates
    start_date = calibration_start_dates[country][area]
    end_date = calibration_lockdown_dates[country]['end']
    measure_start_date = calibration_lockdown_dates[country]['start']
    measure_window_in_hours = dict()
    measure_window_in_hours['start'] = (pd.to_datetime(measure_start_date) - pd.to_datetime(start_date)).days * TO_HOURS
    measure_window_in_hours['end'] = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * TO_HOURS

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
    m = [
        UpperBoundCasesBetaMultiplier(t_window=Interval(measure_window_in_hours['start'], measure_window_in_hours['end']),
                                      beta_multiplier=beta_multiplier,
                                      max_pos_tests_per_week_per_100k=max_pos_tests_per_week_per_100k,
                                      intervention_times=intervention_times,
                                      init_active=is_measure_active_initially),

        UpperBoundCasesSocialDistancing(t_window=Interval(measure_window_in_hours['start'], measure_window_in_hours['end']),
                                        p_stay_home=p_stay_home,
                                        max_pos_tests_per_week_per_100k=max_pos_tests_per_week_per_100k,
                                        intervention_times=intervention_times,
                                        init_active=is_measure_active_initially)
        ]

    simulation_info = options_to_str(
        max_pos_tests_per_week_per_100k=50,
        initially_active=is_measure_active_initially,
        p_stay_home=p_stay_home
    )

    experiment.add(
        simulation_info=simulation_info,
        country=country,
        area=area,
        measure_list=m,
        lockdown_measures_active=False,
        test_update=None,
        seed_summary_path=seed_summary_path,
        set_calibrated_params_to=calibrated_params,
        set_initial_seeds_to=set_initial_seeds_to,
        full_scale=full_scale,
        expected_daily_base_expo_per100k=expected_daily_base_expo_per100k)

    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

