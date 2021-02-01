import sys

from lib import mobility_reduction
from lib.mobility_reduction import get_mobility_reduction

if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
from lib.measures import *
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.calibrationSettings import calibration_lockdown_dates, calibration_start_dates, \
    calibration_lockdown_beta_multipliers, calibration_mobility_reduction
from lib.calibrationFunctions import get_calibrated_params, get_unique_calibration_params

TO_HOURS = 24.0

if __name__ == '__main__':

    # command line parsing
    args = process_command_line()
    cal_country = args.country
    cal_area = args.area
    cpu_count = args.cpu_count
    continued_run = args.continued

    name = 'validation'
    random_repeats = 100
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = None
    condensed_summary = True

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    calibration_regions = [('CH', 'JU'), ('CH', 'BE')]
    assert (cal_country, cal_area) in calibration_regions
    validation_regions = {'JU': [('CH', 'JU'), ('GER', 'SB'), ('GER', 'RH')],
                          'BE': [('CH', 'BE'), ('GER', 'TU'), ('GER', 'KL')]}

    if args.smoke_test:
        random_repeats = 1
        full_scale = False

    for val_country, val_area in validation_regions[cal_area]:
        # set simulation and intervention dates
        start_date = calibration_start_dates[val_country][val_area]
        end_date = calibration_lockdown_dates[val_country]['end']
        measure_start_date = calibration_lockdown_dates[val_country]['start']
        measure_window_in_hours = dict()
        measure_window_in_hours['start'] = (pd.to_datetime(measure_start_date) - pd.to_datetime(
            start_date)).days * TO_HOURS
        measure_window_in_hours['end'] = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * TO_HOURS

        # create experiment object
        experiment_info = f'{name}-{cal_area}'
        experiment = Experiment(
            experiment_info=experiment_info,
            start_date=start_date,
            end_date=end_date,
            random_repeats=random_repeats,
            cpu_count=cpu_count,
            full_scale=full_scale,
            verbose=verbose,
        )

        calibrated_params = get_calibrated_params(country=cal_country, area=cal_area,
                                                  multi_beta_calibration=False,
                                                  estimate_mobility_reduction=False)

        # measures
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        m = [
            SocialDistancingBySiteTypeForAllMeasure(
                    t_window=Interval(
                        measure_window_in_hours['start'],
                        measure_window_in_hours['end']),
                    p_stay_home_dict=calibration_mobility_reduction[val_country][val_area]),

            BetaMultiplierMeasureByType(
                t_window=Interval(
                    measure_window_in_hours['start'],
                    measure_window_in_hours['end']),
                beta_multiplier=calibration_lockdown_beta_multipliers)
            ]

        sim_info = options_to_str(validation_region=val_area)

        experiment.add(
            simulation_info=sim_info,
            country=val_country,
            area=val_area,
            measure_list=m,
            test_update=None,
            seed_summary_path=seed_summary_path,
            set_calibrated_params_to=calibrated_params,
            set_initial_seeds_to=set_initial_seeds_to,
            full_scale=full_scale)

        print(f'{experiment_info} configuration done.')

        # execute all simulations
        experiment.run_all()
