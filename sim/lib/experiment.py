import sys
if '..' not in sys.path:
    sys.path.append('..')

import pickle, multiprocessing, copy
import pandas as pd
import numpy as np
import botorch.utils.transforms as transforms
from lib.calibrationFunctions import (
    pdict_to_parr, parr_to_pdict, save_state, load_state, 
    get_calibrated_params, gen_initial_seeds, get_test_capacity, downsample_cases)
from lib.mobilitysim import MobilitySimulator
from lib.parallel import launch_parallel_simulations
from lib.distributions import CovidDistributions
from lib.data import collect_data_from_df
from lib.measures import *
from lib.calibrationSettings import (calibration_lockdown_dates, calibration_testing_params)

TO_HOURS = 24.0

'''
Helper functions to run experiments
'''

def save_summary(summary, filename):
    '''Saves summary files'''
    with open('summaries/' + filename, 'wb') as fp:
        pickle.dump(summary, fp)


def load_summary(filename):
    '''Loads summary file'''
    with open('summaries/' + filename, 'rb') as fp:
        summary = pickle.load(fp)
    return summary


def run_experiment(country, area, mob_settings, start_date, end_date, random_repeats, measure_list, 
                   test_update=None, seed_summary_path=None, return_mob=False, set_calibrated_params_to=None,
                   multi_beta_calibration=False):

    '''
    Runs experiment for `country` and `area` from a `start_date` until an `end_date`
    given a provided `measure_list`.
    The test parameter dictionary in `calibration_settings.py` can be amended by passing a function `test_update`.\
    '''

    # Set time window based on start and end date
    sim_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    max_time = TO_HOURS * sim_days  # in hours

    # Load mobility object for country + area
    with open(mob_settings, 'rb') as fp:
        obj = pickle.load(fp)
    mob = MobilitySimulator(**obj)
    mob.simulate(max_time=max_time, lazy_contacts=True)

    # Obtain COVID19 case date for country and area to estimate testing capacity and heuristic seeds if necessary
    unscaled_area_cases = collect_data_from_df(country=country, area=area, datatype='new',
                                               start_date_string=start_date, end_date_string=end_date)
    assert(len(unscaled_area_cases.shape) == 2)

    # Scale down cases based on number of people in town, region, and downsampling
    sim_cases = downsample_cases(unscaled_area_cases, mob)

    # Get initial seeds for simulation
    # (a) Define heuristically based on true cases and literature distribution estimates
    if seed_summary_path is None:

        # Generate initial seeds based on unscaled case numbers in town
        initial_seeds = gen_initial_seeds(
            sim_cases, day=0)

        if sum(initial_seeds.values()) == 0:
            print('No states seeded at start time; cannot start simulation.\n'
                'Consider setting a later start date for calibration using the "--start" flag.')
            sys.exit(0)

    # (b) Define based state of previous batch of simulations,
    # using the random rollout that best matched the true cases in terms of squared error
    else:
        seed_summary_ = load_summary(seed_summary_path)
        seed_day_ = seed_summary_.max_time # take seeds at the end of simulation
        initial_seeds = extract_seeds_from_summary(
            seed_summary_, seed_day_, sim_cases)

    # Instantiate correct state transition distributions (estimated from in literature)
    distributions = CovidDistributions(country=country)

    # Add standard measure of positives staying isolated
    measure_list += [
        SocialDistancingForPositiveMeasure(
            t_window=Interval(0.0, max_time), p_stay_home=1.0),

        SocialDistancingForPositiveMeasureHousehold(
            t_window=Interval(0.0, max_time), p_isolate=1.0)
    ]
    measure_list = MeasureList(measure_list)

    # Load calibrated model parameters for this area
    calibrated_params = set_calibrated_params_to or get_calibrated_params(
        country=country, area=area, multi_beta_calibration=multi_beta_calibration)

    if multi_beta_calibration:
        betas = calibrated_params['betas']
    else:
        betas = {
            'education': calibrated_params['beta_site'],
            'social': calibrated_params['beta_site'],
            'bus_stop': calibrated_params['beta_site'],
            'office': calibrated_params['beta_site'],
            'supermarket': calibrated_params['beta_site'],
        }

    model_params = {
        'betas' : betas,
        'beta_household': calibrated_params['beta_household'],
    }

    # Set testing conditions
    scaled_test_capacity = get_test_capacity(country, area, mob, end_date_string=end_date)
    testing_params = copy.deepcopy(calibration_testing_params)
    testing_params['tests_per_batch'] = scaled_test_capacity
    testing_params['testing_t_window'] = [0.0, max_time]
    if test_update:
        testing_params = test_update(testing_params)

    # Run simulations
    summary = launch_parallel_simulations(
        mob_settings=mob_settings,
        distributions=distributions,
        random_repeats=random_repeats,
        cpu_count=multiprocessing.cpu_count(),
        params=model_params,
        initial_seeds=initial_seeds,
        testing_params=testing_params,
        measure_list=measure_list,
        max_time=max_time,
        num_people=mob.num_people,
        num_sites=mob.num_sites,
        site_loc=mob.site_loc,
        home_loc=mob.home_loc,
        lazy_contacts=True,
        verbose=False)

    if return_mob:
        return summary, mob
    else:
        return summary
