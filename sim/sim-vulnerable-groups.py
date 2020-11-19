
import sys
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
from lib.measures import *
from lib.experiment import Experiment, process_command_line
from lib.calibrationFunctions import get_calibrated_params

TO_HOURS = 24.0

if __name__ == '__main__':

    name = 'vulnerable-groups'
    random_repeats = 48
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = None

    # command line parsing
    args = process_command_line()
    config = args.config
    cpu_count = args.cpu_count

    # Load calibrated parameters up to `maxBOiters` iterations of BO
    maxBOiters = 40 if config.area in ['BE', 'JU', 'RH'] else None
    calibrated_params = get_calibrated_params(config=config,
                                              multi_beta_calibration=False,
                                              maxiters=maxBOiters)

    # experiment parameters
    # Isolate older age groups for `weeks` number of weeks
    p_stay_home = calibrated_params['p_stay_home']

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # set simulation and intervention dates
    start_date = config.calibration_start_dates
    end_date = config.calibration_end_dates
    measure_start_date = config.calibration_lockdown_dates['start']
    measure_window_in_hours = dict()
    measure_window_in_hours['start'] = (pd.to_datetime(measure_start_date) - pd.to_datetime(start_date)).days * TO_HOURS
    measure_window_in_hours['end'] = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * TO_HOURS

    # create experiment object
    experiment_info = f'{name}-{config.country}-{config.area}'
    experiment = Experiment(
        experiment_info=experiment_info,
        start_date=start_date,
        end_date=end_date,
        random_repeats=random_repeats,
        cpu_count=cpu_count,
        full_scale=full_scale,
        verbose=verbose,
    )

    # Social distancing for vulnerable people (older age groups) for different time periods
    # measures
    max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    if config.country == 'GER':
        p_stay_home_per_age_group = [0.0, 0.0, 0.0, 0.0, p_stay_home, p_stay_home]
    elif config.country == 'CH':
        p_stay_home_per_age_group = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_stay_home, p_stay_home, p_stay_home]
    else:
        raise NotImplementedError(f'Set p_stay_home_per_age_group according to age groups in the respective country')

    m = [
        SocialDistancingByAgeMeasure(
            t_window=Interval(
                measure_window_in_hours['start'], 
                measure_window_in_hours['end']),
            p_stay_home=p_stay_home_per_age_group)
        ]

    simulation_info = ''

    experiment.add(
        simulation_info=simulation_info,
        config=config,
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

