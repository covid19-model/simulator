
import sys
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
from lib.measures import *
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.calibrationSettings import calibration_lockdown_dates, calibration_start_dates, calibration_lockdown_beta_multipliers
from lib.calibrationFunctions import get_calibrated_params

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
    condensed_summary = True

    # set `True` for narrow-casting plot; should only be done with 1 random restart:
    store_mob = True
    cpu_count = 1
    random_repeats = 1

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # Load calibrated parameters up to `maxBOiters` iterations of BO
    maxBOiters = 40 if area in ['BE', 'JU', 'RH'] else None
    calibrated_params = get_calibrated_params(country=country, area=area,
                                              multi_beta_calibration=False,
                                              maxiters=maxBOiters)

    # experiment parameters
    p_stay_home = calibrated_params['p_stay_home']

    if args.smoke_test:
        start_date = '2021-01-01'
        end_date = '2021-02-15'
        random_repeats = 1
        full_scale = False
        k_groups = [4]

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
        SocialDistancingForAllMeasure(
            t_window=Interval(0.0, TO_HOURS * max_days),
            p_stay_home=p_stay_home),

        BetaMultiplierMeasureByType(
            t_window=Interval(0.0, TO_HOURS * max_days),
            beta_multiplier=calibration_lockdown_beta_multipliers)
        ]

    simulation_info = 'single'

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
        expected_daily_base_expo_per100k=expected_daily_base_expo_per100k,
        store_mob=store_mob)

    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

