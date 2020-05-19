import sys
if '..' not in sys.path:
    sys.path.append('..')

import pickle, multiprocessing, copy
import pandas as pd
import numpy as np
import botorch.utils.transforms as transforms
from lib.inference import (
    pdict_to_parr, parr_to_pdict, save_state, load_state, get_calibrated_params, gen_initial_seeds)
from lib.mobilitysim import MobilitySimulator
from lib.parallel import launch_parallel_simulations
from lib.distributions import CovidDistributions
from lib.data import collect_data_from_df
from lib.measures import *
from lib.calibration_settings import (settings_lockdown_dates, settings_testing_params)

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
                   test_update=None, seed_summary_path=None):

    '''
    Runs experiment for `country` and `area` from a `start_date` until an `end_date`
    given a provided `measure_list`.
    The test parameter dictionary in `calibration_settings.py` can be amended by passing a function `test_update`.\
    '''

    # Load mobility object for country + area
    with open(mob_settings, 'rb') as fp:
        obj = pickle.load(fp)
    mob = MobilitySimulator(**obj)

    # Set time window based on start and end date
    sim_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    max_time = TO_HOURS * sim_days  # in hours

    # Obtain COVID19 case date for country and area to estimate testing capacity and heuristic seeds in necessary
    new_cases_ = collect_data_from_df(country=country, area=area, datatype='new',
                                      start_date_string=start_date, end_date_string=end_date)

    new_cases = np.ceil(
        (new_cases_ * mob.num_people_unscaled) /
        (mob.downsample * mob.region_population))

    # Get initial seeds for simulation
    # (a) Define heuristically based on true cases and literature distribution estimates
    if seed_summary_path is None:
        initial_seeds = gen_initial_seeds(new_cases)

    # (b) Define based state of previous batch of simulations,
    # using the random rollout that best matched the true cases in terms of squared error
    else:
        seed_summary_ = load_summary(seed_summary_path)
        seed_day_ = seed_summary_.max_time # take seeds at the end of simulaiton
        initial_seeds = extract_seeds_from_summary(
            seed_summary_, seed_day_, new_cases)

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
    calibrated_params = get_calibrated_params(country, area)

    # Set testing conditions
    daily_case_increase = new_cases.sum(axis=1)[1:] - new_cases.sum(axis=1)[:-1]
    testing_params = copy.deepcopy(settings_testing_params)
    testing_params['tests_per_batch'] = int(daily_case_increase.max())
    testing_params['testing_t_window'] = [0.0, max_time]
    if test_update:
        testing_params = test_update(testing_params)

    # Run simulations
    summary = launch_parallel_simulations(
        mob_settings=mob_settings,
        distributions=distributions,
        random_repeats=random_repeats,
        cpu_count=multiprocessing.cpu_count(),
        params=calibrated_params,
        initial_seeds=initial_seeds,
        testing_params=testing_params,
        measure_list=measure_list,
        max_time=max_time,
        num_people=mob.num_people,
        num_sites=mob.num_sites,
        site_loc=mob.site_loc,
        home_loc=mob.home_loc,
        dynamic_tracing=True,
        verbose=False)
    return summary
