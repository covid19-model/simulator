
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

    name = 'k-groups'
    random_repeats = 96
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = None

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
    # Split citizens in `K_groups` groups and alternatingly install social distancing measures for the groups
    # `K_groups_weeks` determines for how many weeks this strategy is active
    K_groups = [2, 3, 4]

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

    # Isolate k groups for different numbers of groups
    for groups in K_groups:
        # measures
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        m = [
            SocialDistancingForKGroups(
                t_window=Interval(measure_window_in_hours['start'], TO_HOURS * 7 * K_groups_weeks),
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
            lockdown_measures_active=False,
            test_update=None,
            seed_summary_path=seed_summary_path,
            set_calibrated_params_to=calibrated_params,
            set_initial_seeds_to=set_initial_seeds_to,
            full_scale=full_scale)
    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

