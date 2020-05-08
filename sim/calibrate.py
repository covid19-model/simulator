from datetime import datetime
import sys
import argparse
if '..' not in sys.path:
    sys.path.append('..')

import pandas as pd
import numpy as np
import networkx as nx
import copy
import scipy as sp
import math
import seaborn
import pickle
import warnings
import matplotlib
import re
import multiprocessing
import torch

from botorch import fit_gpytorch_model
from botorch.exceptions import BadInitialCandidatesWarning
import botorch.utils.transforms as transforms
from lib.inference import make_bayes_opt_functions, pdict_to_parr, parr_to_pdict, InferenceLogger, save_state, load_state, gen_initial_seeds
from lib.inference_kg import qKnowledgeGradient
import time, pprint

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from lib.mobilitysim import MobilitySimulator
from lib.dynamics import DiseaseModel
from bayes_opt import BayesianOptimization
from lib.parallel import *
from lib.distributions import CovidDistributions
from lib.plot import Plotter
from lib.data import collect_data_from_df
from lib.measures import (
    MeasureList, 
    SocialDistancingForAllMeasure, 
    SocialDistancingByAgeMeasure,
    SocialDistancingForPositiveMeasure,
    SocialDistancingForPositiveMeasureHousehold,
    Interval)

from lib.mobilitysim import MobilitySimulator

from lib.settings.calibration_settings import *

if __name__ == '__main__':

    '''
    Command line arguments
    '''

    # command line arguments change the standard settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="set seed")

    # BO
    parser.add_argument("--ninit", type=int, help="update default number of quasi-random initial evaluations")
    parser.add_argument("--niters", type=int, help="update default number of BO iterations")
    parser.add_argument("--rollouts", type=int, help="update default number of parallel simulation rollouts")
    parser.add_argument("--load", help="specify path to a BO state to be loaded as initial observations, e.g. 'logs/calibration_0_state.pk'")

    # data
    parser.add_argument("--mob", help="specify path to mobility settings for trace generation, e.g. 'lib/tu_settings_10_10_hh.pk'")
    parser.add_argument("--country", help="specify country indicator for data import")
    parser.add_argument("--area", help="specify area indicator for data import")
    parser.add_argument("--days", type=int, help="specify number of days for which case data is retrieved")
    parser.add_argument("--start", help="adjust starting data for which case data is retrieved"
                        "default for 'GER' should be '2020-03-10'")
    parser.add_argument("--downsample", type=int, help="update default case downsampling factor")

    # simulation
    parser.add_argument("--endsimat", type=int, help="for debugging: specify number of days after which simulation should be cut off")
    parser.add_argument("--testingcap", type=int, help="update default unscaled testing capacity")

    
    # Read arguments from the command line
    args = parser.parse_args()

    if args.seed:
        seed = args.seed
    else:
        seed = 0
    
    if args.mob:
        mob_settings = args.mob
    else:
        print("Need to set path to mobility settings for trace generation, \ne.g. python calibrate.py --mob \"lib/tu_settings_10_10_hh.pk\"")
        exit(0)

    if args.area and args.country and args.days:
        data_area = args.area
        data_country = args.country
        data_days = args.days
    else:
        print("Need to set days, country, and area identifier for data\ne.g. python calibrate.py --country \"GER\" --area \"TU\" --days 16")
        exit(0)

    if args.downsample:
        case_downsampling = args.downsample
    else:
        print("Need to set downsampling factor, e.g. '--downsample 5', also in correspondance with --mob")
        exit(0)

    """
    All settings should be changed in `./settings/calibration_settings.py`
    or passed via command line
    """

    initial_lines_printed = []

    # remember line executed
    initial_lines_printed.append('=' * 100)
    initial_lines_printed.append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    initial_lines_printed.append('python ' + ' '.join(sys.argv))
    initial_lines_printed.append('=' * 100)

    # data settings
    filename = f'calibration_{seed}'
    verbose = settings_data['verbose']
    use_households = settings_data['use_households']
    data_start_date = args.start or settings_data['data_start_date']
    debug_simulation_days = args.endsimat # if not None, simulation will be cut short for debugging

    # initialize mobility object to obtain information (no trace generation yet)
    with open(mob_settings, 'rb') as fp:
        kwargs = pickle.load(fp)
    mob = MobilitySimulator(**kwargs)

    # number of tests processed every `testing_frequency` hours
    unscaled_testing_capacity = args.testingcap or mob.daily_tests_per_100k # FIXME: better name of field: `mob.daily_tests`
    population_unscaled = mob.num_people_unscaled 
    
    # simulation settings
    n_init_samples = args.ninit or settings_simulation['n_init_samples']
    n_iterations = args.niters or settings_simulation['n_iterations']
    simulation_roll_outs = args.rollouts or settings_simulation['simulation_roll_outs']
    cpu_count = settings_simulation['cpu_count']
    dynamic_tracing = True
    load_observations = args.load

    # parameter bounds
    param_bounds = settings_param_bounds 

    # set testing parameters
    testing_params = settings_testing_params

    # BO acquisition function optimization (Knowledge gradient)
    acqf_opt_num_fantasies = settings_acqf['acqf_opt_num_fantasies']
    acqf_opt_num_restarts = settings_acqf['acqf_opt_num_restarts']
    acqf_opt_raw_samples = settings_acqf['acqf_opt_raw_samples']
    acqf_opt_batch_limit = settings_acqf['acqf_opt_batch_limit']
    acqf_opt_maxiter = settings_acqf['acqf_opt_maxiter']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Bayesian optimization pipeline
    """

    # Based on population size, approx. 300 tests/day in Area of Tübingen (~135 in city of Tübingen)
    tests_per_day = math.ceil(unscaled_testing_capacity / case_downsampling)

    # set testing parameters
    testing_params['tests_per_batch'] = tests_per_day

    test_lag_days = int(testing_params['test_reporting_lag'] / 24.0)
    assert(int(testing_params['test_reporting_lag']) % 24 == 0)

    # Import Covid19 data
    new_cases_ = collect_data_from_df(country=data_country, area=data_area, datatype='new', 
        start_date_string=data_start_date, days=data_days)
    resistant_cases_ = collect_data_from_df(country=data_country, area=data_area, datatype='recovered', 
        start_date_string=data_start_date, days=data_days)
    fatality_cases_ = collect_data_from_df(country=data_country, area=data_area, datatype='fatality', 
        start_date_string=data_start_date, days=data_days)

    assert(len(new_cases_.shape) == 2 and len(resistant_cases_.shape) == 2 and len(fatality_cases_.shape) == 2)

    if new_cases_[0].sum() == 0:
        print('No positive cases at provided start time; cannot seed simulation.\n'
              'Consider setting a later start date for calibration using the "--start" flag.')
        exit(0)

    
    # Empirical fatality rate per age group from the above data. 
    # RKI data defines 6 groups: **0-4y, 5-14y, 15-34y, 35-59y, 60-79y, 80+y**
    num_age_groups = fatality_cases_.shape[1] 
    fatality_rates_by_age = (fatality_cases_[-1, :] / 
        (new_cases_[-1, :] +  fatality_cases_[-1, :] + resistant_cases_[-1, :]))
    fatality_rates_by_age = np.nan_to_num(fatality_rates_by_age) # deal with 0/0

    # Scale down cases based on number of people in simulation
    new_cases, resistant_cases, fatality_cases = (
        np.ceil(1/case_downsampling * new_cases_),
        np.ceil(1/case_downsampling * resistant_cases_),
        np.ceil(1/case_downsampling * fatality_cases_))

    # generate initial seeds based on case numbers
    initial_seeds = gen_initial_seeds(new_cases)
    initial_lines_printed.append('Initial seed counts : ' + str(initial_seeds))

    # in debug mode, shorten time of simulation, shorten time
    if debug_simulation_days:
        new_cases = new_cases[:debug_simulation_days]

    # Maximum time fixed by real data, init mobility simulator simulation
    max_time = int(new_cases.shape[0] * 24.0) # maximum time to simulate, in hours
    max_time += 24.0 * test_lag_days # longer due to test lag in simulations
    testing_params['testing_t_window'] = [0.0, max_time]
    mob.simulate(max_time=max_time, dynamic_tracing=True)


    initial_lines_printed.append(
        'Max time T (days): ' + str(new_cases.shape[0]))
    initial_lines_printed.append(
        'Target cases per age group at t=0:   ' + str(list(map(int, new_cases[0].tolist()))))
    initial_lines_printed.append(
        'Target cases per age group at t=T:   ' + str(list(map(int, new_cases[-1].tolist()))))

    # instantiate correct distributions
    distributions = CovidDistributions(fatality_rates_by_age=fatality_rates_by_age)

    # standard quarantine of positive tests staying at home in isolation
    measure_list = MeasureList([
        SocialDistancingForPositiveMeasure(
            t_window=Interval(0.0, max_time), p_stay_home=1.0),
        SocialDistancingForPositiveMeasureHousehold(
            t_window=Interval(0.0, max_time), p_isolate=1.0),
    ])

    # set Bayesian optimization target as positive cases
    n_days, n_age = new_cases.shape
    G_obs = torch.tensor(new_cases).reshape(n_days * n_age) # flattened

    sim_bounds = pdict_to_parr(param_bounds)
    n_params = sim_bounds.shape[0]

    initial_lines_printed.append(f'Parameters : {n_params}')
    initial_lines_printed.append('Parameter bounds: ' + str(parr_to_pdict(sim_bounds)))

    # create settings dictionary for simulations
    launch_kwargs = dict(
        mob_settings=mob_settings,
        distributions=distributions,
        random_repeats=simulation_roll_outs,
        cpu_count=cpu_count,
        initial_seeds=initial_seeds,
        testing_params=testing_params,
        measure_list=measure_list,
        max_time=max_time,
        num_people=mob.num_people,
        num_sites=mob.num_sites,
        home_loc=mob.home_loc,
        site_loc=mob.site_loc,
        dynamic_tracing=dynamic_tracing,
        verbose=False)

    # genereate essential functions for Bayesian optimization
    (objective, 
    generate_initial_observations, 
    initialize_model, 
    optimize_acqf_and_get_observation) = \
        make_bayes_opt_functions(
            targets_cumulative=new_cases,
            n_params=n_params, 
            n_days=n_days, 
            n_age=n_age, 
            sim_bounds=sim_bounds.T, 
            test_lag_days=test_lag_days, 
            launch_kwargs=launch_kwargs, 
            verbose=verbose)

    initial_lines_printed.append('Negative iteration indices indicate initial quasi-random exploration.')
    initial_lines_printed.append('`diff` indicates `total sim cases at t=T - total true cases at t=T`')
    initial_lines_printed.append('`walltime` indicates time in minutes needed to perform iteration')

    # generate initial training data (either load or simulate)
    if load_observations:

        # load initial observations 
        state = load_state(load_observations)
        train_theta = state['train_theta']
        train_G = state['train_G']
        train_G_sem = state['train_G_sem']
        best_observed_obj = state['best_observed_obj']
        best_observed_idx = state['best_observed_idx']

        print('Loaded initial observations from ' + load_observations)
        print(f'Observations: {train_theta.shape[0]}, Best objective: {best_observed_obj}')

        logger = InferenceLogger(
            filename=filename, initial_lines=initial_lines_printed, verbose=verbose)

    else:

        logger = InferenceLogger(
            filename=filename, initial_lines=initial_lines_printed, verbose=verbose)

        # generate initial training data
        train_theta, train_G, train_G_sem, best_observed_obj, best_observed_idx = generate_initial_observations(
            n=n_init_samples, logger=logger)

    # init model based on initial observations
    mll, model = initialize_model(train_theta, train_G, train_G_sem)

    best_observed = []
    best_observed.append(best_observed_obj)

    # run n_iterations rounds of BayesOpt after the initial random batch
    for tt in range(n_iterations):
        
        t0 = time.time()

        # fit the GP model
        fit_gpytorch_model(mll)

        # define acquisition function based on fitted GP
        acqf = qKnowledgeGradient(
            model=model,
            objective=objective,
            num_fantasies=acqf_opt_num_fantasies,
        )
        
        # optimize acquisition and get new observation via simulation at selected parameters
        new_theta, new_G, new_G_sem = optimize_acqf_and_get_observation(
            acq_func=acqf,
            acqf_opt_num_restarts=acqf_opt_num_restarts, 
            acqf_opt_raw_samples=acqf_opt_raw_samples, 
            acqf_opt_batch_limit=acqf_opt_batch_limit, 
            acqf_opt_maxiter=acqf_opt_maxiter)
            
        # concatenate observations
        train_theta = torch.cat([train_theta, new_theta], dim=0) 
        train_G = torch.cat([train_G, new_G], dim=0) 
        train_G_sem = torch.cat([train_G_sem, new_G_sem], dim=0) 
        
        # update progress
        train_G_objectives = objective(train_G)
        best_observed_idx = train_G_objectives.argmax()
        best_observed_obj = train_G_objectives[best_observed_idx].item()
        best_observed.append(best_observed_obj)
        
        # re-initialize the models so they are ready for fitting on next iteration
        mll, model = initialize_model(
            train_theta, 
            train_G, 
            train_G_sem,
        )

        t1 = time.time()
        
        # log
        case_diff = (
            new_G.reshape(n_days, n_age)[-1].sum() 
            - torch.tensor(new_cases)[-1].sum())
        
        logger.log(
            i=tt,
            time=t1 - t0,
            best=best_observed_obj,
            case_diff=case_diff,
            objective=objective(new_G).item(),
            theta=transforms.unnormalize(new_theta.detach().squeeze(), sim_bounds.T)
        )

        # save state
        state = {
            'train_theta' : train_theta,
            'train_G' : train_G,
            'train_G_sem'  : train_G_sem,
            'best_observed_obj': best_observed_obj,
            'best_observed_idx': best_observed_idx
        }
        save_state(state, filename)

    # print best parameters
    print()
    print('FINISHED.')
    print('Best objective:  ', best_observed_obj)
    print('Best parameters:')
    
    # scale back to simulation parameters (from unit cube parameters in BO)
    normalized_calibrated_params = train_theta[best_observed_idx]
    calibrated_params = transforms.unnormalize(normalized_calibrated_params, sim_bounds.T)
    pprint.pprint(parr_to_pdict(calibrated_params))


