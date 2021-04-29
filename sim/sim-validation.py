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
    calibration_lockdown_beta_multipliers, calibration_mobility_reduction, calibration_lockdown_site_closures
from lib.calibrationFunctions import get_calibrated_params, get_unique_calibration_params, \
    get_calibrated_params_from_path

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

    calibration_regions = [('CH', 'JU'), ('CH', 'BE'), ('GER', 'TU'), ('GER', 'KL'), ('GER', 'RH')]
    assert (cal_country, cal_area) in calibration_regions
    validation_regions = {
        'BE': [('CH', 'BE')],
        'JU': [('CH', 'JU')],
        'TU': [('GER', 'TU')],
        'KL': [('GER', 'KL')],
        'RH': [('GER', 'RH')],
    }
       
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

        if not args.calibration_state:
            calibrated_params = get_calibrated_params(country=cal_country, area=cal_area)
        else:
            calibrated_params = get_calibrated_params_from_path(args.calibration_state)
            print('Loaded non-standard calibration state.')

        # measures
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        p_stay_home_dict_mobility_reduced = calibration_mobility_reduction[val_country][val_area]
        p_stay_home_dict_closures = {site_type: 1.0 for site_type in calibration_lockdown_site_closures}
        p_stay_home_dict = {**p_stay_home_dict_closures, **p_stay_home_dict_mobility_reduced}

        m = [
            SocialDistancingBySiteTypeForAllMeasure(
                    t_window=Interval(
                        measure_window_in_hours['start'],
                        measure_window_in_hours['end']),
                    p_stay_home_dict=p_stay_home_dict),
            ]

        sim_info = options_to_str(validation_region=val_area)

        experiment.add(
            simulation_info=sim_info,
            country=val_country,
            area=val_area,
            measure_list=m,
            seed_summary_path=seed_summary_path,
            set_calibrated_params_to=calibrated_params,
            set_initial_seeds_to=set_initial_seeds_to,
            full_scale=full_scale)

        print(f'{experiment_info} configuration done.')

        # execute all simulations
        experiment.run_all()
