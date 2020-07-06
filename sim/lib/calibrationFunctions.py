import time
import os
import sys
import asyncio
import threading
import json
import pprint
import csv
from datetime import datetime, timedelta

from lib.priorityqueue import PriorityQueue
from lib.dynamics import DiseaseModel
from lib.mobilitysim import MobilitySimulator
from lib.parallel import *

import gpytorch, torch, botorch, sobol_seq, pandas
from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize
from botorch.models import FixedNoiseGP, ModelListGP, HeteroskedasticSingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood
from botorch.acquisition.monte_carlo import MCAcquisitionFunction, qNoisyExpectedImprovement, qSimpleRegret
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition import OneShotAcquisitionFunction
import botorch.utils.transforms as transforms
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform

from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import GenericMCObjective, ConstrainedMCObjective
from botorch.gen import get_best_candidates, gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions

from lib.kg import qKnowledgeGradient, gen_one_shot_kg_initial_conditions
from lib.distributions import CovidDistributions
from lib.calibrationSettings import (
    calibration_model_param_bounds_single, 
    calibration_model_param_bounds_multi, 
    calibration_testing_params,
    calibration_lockdown_dates,
    calibration_states,
    calibration_mob_paths,
    calibration_start_dates,
    calibration_lockdown_beta_multipliers
)

from lib.data import collect_data_from_df

from lib.measures import (
    MeasureList,
    SocialDistancingForAllMeasure,
    SocialDistancingByAgeMeasure,
    SocialDistancingForPositiveMeasure,
    SocialDistancingForPositiveMeasureHousehold,
    Interval)


import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MIN_NOISE = torch.tensor(1e-6)
TO_HOURS = 24.0

class CalibrationLogger:

    def __init__(
        self,
        filename,
        multi_beta_calibration,
        verbose
    ):

        self.dir = 'logs/'
        self.filename = filename
        self.multi_beta_calibration = multi_beta_calibration

        if multi_beta_calibration:
            self.headers = [
                'iter',
                '    best obj',
                ' current obj',
                ' diff',
                'b/educat',
                'b/social',
                'b/bus_st',
                'b/office',
                'b/superm',
                'b/househ',
                '  p_home',
                'walltime',
            ]
        else:
            self.headers = [
                'iter',
                '    best obj',
                ' current obj',
                ' diff',
                '  b/site',
                'b/househ',
                '  p_home',
                'walltime',
            ]

        self.verbose = verbose

    def log_initial_lines(self, initial_lines):
        '''
        Writes `initial_lines` to top of log file.
        '''

        self.initial_lines = initial_lines

        # write headers
        with open(f'{self.dir + self.filename}.csv', 'w+') as logfile:

            wr = csv.writer(logfile, quoting=csv.QUOTE_ALL)
            for l in self.initial_lines:
                wr.writerow([l])
            wr.writerow([""])
            wr.writerow(self.headers)

        # print to stdout if verbose
        if self.verbose:
            for l in self.initial_lines:
                print(l)
            print()
            headerstrg = ' | '.join(self.headers)
            print(headerstrg)

    def log(self, i, time, best, objective, case_diff, theta):
        '''
        Writes lst to a .csv file
        '''
        d = parr_to_pdict(parr=theta, multi_beta_calibration=self.multi_beta_calibration)
        fields = [
            f"{i:4.0f}",
            f"{best:12.4f}",
            f"{objective:12.4f}",
            f"{case_diff:5.0f}",
        ]
        if self.multi_beta_calibration:
            fields += [
                f"{d['betas']['education']:8.4f}",
                f"{d['betas']['social']:8.4f}",
                f"{d['betas']['bus_stop']:8.4f}",
                f"{d['betas']['office']:8.4f}",
                f"{d['betas']['supermarket']:8.4f}",
            ]
        else:
            fields += [
                f"{d['beta_site']:8.4f}",
            ]

        fields += [
            f"{d['beta_household']:8.4f}",
            f"{d['p_stay_home']:8.4f}",
            f"{time/60.0:8.4f}",
        ]

        with open(f'{self.dir + self.filename}.csv', 'a') as logfile:

            wr = csv.writer(logfile, quoting=csv.QUOTE_ALL)
            wr.writerow(fields)

        # print to stdout if verbose
        if self.verbose:
            outstrg = ' | '.join(list(map(str, fields)))
            print(outstrg)

        return

def extract_seeds_from_summary(summary, t, real_cases):
    '''
    Extracts initial simulation seeds from a summary file at time `t` 
    based on lowest objective value of run, i.e. lowest squared error per age group over time
    '''
    calib_legal_states = ['susc', 'expo', 'ipre', 'isym',
                          'iasy', 'posi', 'nega', 'resi', 'dead', 'hosp']

    real_cases = torch.tensor(real_cases)

    # summary into cumulative daily positives cases
    cumulative = convert_timings_to_cumulative_daily(
        torch.tensor(summary.state_started_at['posi']), 
        torch.tensor(summary.people_age), 
        real_cases.shape[0] * TO_HOURS)

    # objectives per random restart
    # squared error (in aggregate, i.e. summed over age group before computing squared difference)
    objectives = (cumulative.sum(dim=-1) - real_cases.unsqueeze(0).sum(dim=-1)).pow(2).sum(dim=-1)
    best = objectives.argmin()

    # compute all states of best run at time t
    states = {}
    for state in calib_legal_states:
        states[state] = (summary.state_started_at[state][best] <= t) \
            & (t < summary.state_ended_at[state][best])
        
    # compute counts (resistant also contain dead)
    expo = states['expo'].sum()
    iasy = states['iasy'].sum()
    ipre = states['ipre'].sum()
    isym_posi = (states['isym'] & states['posi']).sum()
    isym_notposi = (states['isym'] & (1 - states['posi'])).sum()
    resi_posi = ((states['resi'] | states['dead']) & states['posi']).sum()
    resi_notposi = ((states['resi'] | states['dead']) & (1 - states['posi'])).sum()

    seeds = {
        'expo' : int(expo),
        'iasy' : int(iasy),
        'ipre' : int(ipre),
        'isym_posi': int(isym_posi),
        'isym_notposi': int(isym_notposi),
        'resi_posi': int(resi_posi),
        'resi_notposi': int(resi_notposi),
    }
    return seeds

def save_state(obj, filename):
    """Saves `obj` to `filename`"""
    with open('logs/' + filename + '_state.pk', 'wb') as fp:
        torch.save(obj, fp)
    return

def load_state(filename):
    """Loads obj from `filename`"""
    with open(filename, 'rb') as fp:
        obj = torch.load(fp)
    return obj

def pdict_to_parr(*, pdict, multi_beta_calibration):
    """Convert parameter dict to BO parameter tensor"""
    if multi_beta_calibration:
        parr = torch.stack([
            torch.tensor(pdict['betas']['education']),
            torch.tensor(pdict['betas']['social']),
            torch.tensor(pdict['betas']['bus_stop']),
            torch.tensor(pdict['betas']['office']),
            torch.tensor(pdict['betas']['supermarket']),
            torch.tensor(pdict['beta_household']),
            torch.tensor(pdict['p_stay_home']),
        ])
    else:
        parr = torch.stack([
            torch.tensor(pdict['beta_site']),
            torch.tensor(pdict['beta_household']),
            torch.tensor(pdict['p_stay_home']),
        ])
    return parr


def parr_to_pdict(*, parr, multi_beta_calibration):
    """Convert BO parameter tensor to parameter dict"""
    if multi_beta_calibration:
        pdict = {
            'betas': {
                'education': parr[0].tolist(),
                'social': parr[1].tolist(),
                'bus_stop': parr[2].tolist(),
                'office': parr[3].tolist(),
                'supermarket': parr[4].tolist(),
            },
            'beta_household': parr[5].tolist(),
            'p_stay_home': parr[6].tolist(),
        }
    else:
        pdict = {
            'beta_site': parr[0].tolist(),
            'beta_household': parr[1].tolist(),
            'p_stay_home': parr[2].tolist(),
        }
    return pdict


def get_calibrated_params(*, country, area, multi_beta_calibration, maxiters=None):
    """
    Returns calibrated parameters for a `country` and an `area`
    """

    if maxiters:
        param_dict = get_calibrated_params_limited_iters(country, area,
                                                         multi_beta_calibration=multi_beta_calibration,
                                                         maxiters=maxiters,)
        return param_dict

    state = load_state(calibration_states[country][area])
    theta = state['train_theta']
    best_observed_idx = state['best_observed_idx']
    norm_params = theta[best_observed_idx]
    param_bounds = (
        calibration_model_param_bounds_multi
        if multi_beta_calibration else 
        calibration_model_param_bounds_single)
    sim_bounds = pdict_to_parr(
        pdict=param_bounds, multi_beta_calibration=multi_beta_calibration).T
    params = transforms.unnormalize(norm_params, sim_bounds)
    param_dict = parr_to_pdict(parr=params, multi_beta_calibration=multi_beta_calibration)
    return param_dict


def get_calibrated_params_limited_iters(country, area, multi_beta_calibration,  maxiters):
    """
    Returns calibrated parameters using only the first `maxiters` iterations of BO.
    """

    state = load_state(calibration_states[country][area])
    train_G = state['train_G']
    train_G = train_G[:min(maxiters, len(train_G))]
    train_theta = state['train_theta']

    mob_settings = calibration_mob_paths[country][area][0]
    with open(mob_settings, 'rb') as fp:
        mob_kwargs = pickle.load(fp)
    mob = MobilitySimulator(**mob_kwargs)

    data_start_date = calibration_start_dates[country][area]
    data_end_date = calibration_lockdown_dates[country]['end']

    unscaled_area_cases = collect_data_from_df(country=country, area=area, datatype='new',
                                               start_date_string=data_start_date, end_date_string=data_end_date)
    assert (len(unscaled_area_cases.shape) == 2)

    # Scale down cases based on number of people in town and region
    sim_cases = downsample_cases(unscaled_area_cases, mob_kwargs)
    n_days, n_age = sim_cases.shape

    G_obs = torch.tensor(sim_cases).reshape(1, n_days * n_age)
    G_obs_aggregate = torch.tensor(sim_cases).sum(dim=-1)

    def objective(G):
        return - (G - G_obs_aggregate).pow(2).sum(dim=-1) / n_days

    train_G_objectives = objective(train_G)
    best_observed_idx = train_G_objectives.argmax()
    best_observed_obj = train_G_objectives[best_observed_idx].item()

    param_bounds = (
        calibration_model_param_bounds_multi
        if multi_beta_calibration else
        calibration_model_param_bounds_single)
    sim_bounds = pdict_to_parr(
        pdict=param_bounds,
        multi_beta_calibration=multi_beta_calibration
    ).T

    normalized_calibrated_params = train_theta[best_observed_idx]
    calibrated_params = transforms.unnormalize(normalized_calibrated_params, sim_bounds)
    calibrated_params = parr_to_pdict(parr=calibrated_params, multi_beta_calibration=multi_beta_calibration)
    return calibrated_params


def downsample_cases(unscaled_area_cases, mob_settings):
    """
    Generates downsampled case counts based on town and area for a given 2d `cases` array.
    Scaled case count in age group a at time t is

    scaled[t, a] = cases-area[t, a] * (town population / area population)

    """

    unscaled_sim_cases = np.round(unscaled_area_cases * \
        (mob_settings['num_people_unscaled'] / mob_settings['region_population']))
    
    return unscaled_sim_cases


def gen_initial_seeds(unscaled_new_cases, day=0):
    """
    Generates initial seed counts based on unscaled case counts `unscaled_new_cases`.
    The 2d np.array `unscaled_new_cases` has to have shape (num_days, num_age_groups). 

    Assumptions:
    - Cases on day `day` set to number of symptomatic `isym` and positively tested
    - Following literature, asyptomatic indiviudals `iasy` make out approx `alpha` percent of all symtomatics
    - Following literature on R0, set `expo` = R0 * (`isym` + `iasy`)
    - Recovered cases are also considered
    - All other seeds are omitted
    """

    num_days, num_age_groups = unscaled_new_cases.shape

    # set initial seed count (approximately based on infection counts on March 10)
    dists = CovidDistributions(country='GER') # country doesn't matter here
    alpha = dists.alpha
    isym = unscaled_new_cases[day].sum()
    iasy = alpha / (1 - alpha) * isym
    expo = dists.R0 * (isym + iasy)

    seed_counts = {
        'expo': int(np.round(expo).item()),
        'isym_posi': int(np.round(isym).item()),
        'iasy': int(np.round(iasy).item()),
    }
    return seed_counts


def get_test_capacity(country, area, mob_settings, end_date_string='2021-01-01'):
    '''
    Computes heuristic test capacity in `country` and `area` based
    on true case data by determining the maximum daily increase
    in positive cases.
    '''

    unscaled_area_cases = collect_data_from_df(
        country=country, area=area, datatype='new',
        start_date_string='2020-01-01', end_date_string=end_date_string)

    sim_cases = downsample_cases(unscaled_area_cases, mob_settings)

    daily_increase = sim_cases.sum(axis=1)[1:] - sim_cases.sum(axis=1)[:-1]
    test_capacity = int(np.round(daily_increase.max()))
    return test_capacity


def get_scaled_test_threshold(threshold_tests_per_100k, mob):
    '''
    Computes scaled test threshold for conditional measures concept
    '''
    return int(threshold_tests_per_100k / 100000 * mob.num_people)


def convert_timings_to_cumulative_daily(timings, age_groups, time_horizon):
    '''

    Converts batch of size N of timings of M individuals of M age indicators `age_groups` in a time horizon 
    of `time_horizon` in hours into daily cumulative aggregate cases 

    Argument:
        timings :   np.array of shape (N, M)
        age_groups: np.array of shape (N, M)

    Returns:
        timings :   np.array of shape (N, T / 24, `number of age groups`)
    '''
    if len(timings.shape) == 1:
        timings = np.expand_dims(timings, axis=0)

    num_age_groups = torch.unique(age_groups).shape[0]

    # cumulative: (N, T // 24, num_age_groups)
    cumulative = torch.zeros((timings.shape[0], int(time_horizon // 24), num_age_groups))
    for t in range(0, int(time_horizon // 24)):
        for a in range(num_age_groups):
            cumulative[:, t, a] = torch.sum(((timings < (t + 1) * 24) & (age_groups == a)), dim=1)

    return cumulative

def make_bayes_opt_functions(args): 
    '''
    Generates and returns functions used to run Bayesian optimization
    Argument:
        args:                   Keyword arguments specifying exact settings for optimization

    Returns:
        objective :                         objective maximized for BO
        generate_initial_observations :     function to generate initial observations
        initialize_model :                  function to initialize GP
        optimize_acqf_and_get_observation : function to optimize acquisition function based on model
        case_diff :                         computes case difference between prediction array and ground truth at t=T
        unnormalize_theta :                 converts BO params to simulation params (unit cube to real parameters)
        header :                            header lines to be printed to log file

    '''
    header = []

    # set parameter bounds based on calibration mode (single beta vs multiple beta)
    multi_beta_calibration = args.multi_beta_calibration
    if multi_beta_calibration:
        param_bounds = calibration_model_param_bounds_multi
    else:
        param_bounds = calibration_model_param_bounds_single
        
    # remember line executed
    header.append('=' * 100)
    header.append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    header.append('python ' + ' '.join(sys.argv))
    header.append('=' * 100)

    data_country = args.country
    data_area = args.area
    mob_settings = args.mob or calibration_mob_paths[data_country][data_area][0] # 0: scaled; 1: unscaled

    # initialize mobility object to obtain information (no trace generation yet)
    with open(mob_settings, 'rb') as fp:
        mob_kwargs = pickle.load(fp)
    mob = MobilitySimulator(**mob_kwargs)
    
    # data settings
    verbose = not args.not_verbose
    use_households = not args.no_households
    data_start_date = args.start or calibration_start_dates[data_country][data_area]
    data_end_date = args.end or calibration_lockdown_dates[args.country]['end']
    per_age_group_objective = args.per_age_group_objective

    # simulation settings
    n_init_samples = args.ninit
    n_iterations = args.niters
    simulation_roll_outs = args.rollouts
    cpu_count = args.cpu_count
    lazy_contacts = not args.no_lazy_contacts
    load_observations = args.load

    # set testing parameters
    testing_params = calibration_testing_params

    # BO acquisition function optimization (Knowledge gradient)
    acqf_opt_num_fantasies = args.acqf_opt_num_fantasies
    acqf_opt_num_restarts = args.acqf_opt_num_restarts
    acqf_opt_raw_samples = args.acqf_opt_raw_samples
    acqf_opt_batch_limit = args.acqf_opt_batch_limit
    acqf_opt_maxiter = args.acqf_opt_maxiter

    """
    Bayesian optimization pipeline
    """

    # Import Covid19 data
    # Shape (max_days, num_age_groups)
    unscaled_area_cases = collect_data_from_df(country=data_country, area=data_area, datatype='new',
                                               start_date_string=data_start_date, end_date_string=data_end_date)
    assert(len(unscaled_area_cases.shape) == 2)

    # Scale down cases based on number of people in town and region
    sim_cases = downsample_cases(unscaled_area_cases, mob_kwargs)

    # Generate initial seeds based on unscaled case numbers in town
    initial_seeds = gen_initial_seeds(
        sim_cases, day=0)

    if sum(initial_seeds.values()) == 0:
        print('No states seeded at start time; cannot start simulation.\n'
              'Consider setting a later start date for calibration using the "--start" flag.')
        exit(0)

    num_age_groups = sim_cases.shape[1]
    header.append('Downsampling :                    {}'.format(mob.downsample))
    header.append('Simulation population:            {}'.format(mob.num_people))
    header.append('Simulation population (unscaled): {}'.format(mob.num_people_unscaled))
    header.append('Area population :                 {}'.format(mob.region_population))
    header.append('Initial seed counts :             {}'.format(initial_seeds))

    scaled_test_capacity = get_test_capacity(
        country=data_country, area=data_area, 
        mob_settings=mob_kwargs, end_date_string=data_end_date)

    testing_params['tests_per_batch'] = scaled_test_capacity

    test_lag_days = int(testing_params['test_reporting_lag'] / TO_HOURS)
    assert(int(testing_params['test_reporting_lag']) % 24 == 0)

    # Maximum time fixed by real data, init mobility simulator simulation
    # maximum time to simulate, in hours
    max_time = int(sim_cases.shape[0] * TO_HOURS)
    max_time += TO_HOURS * test_lag_days  # simulate longer due to test lag in simulations
    testing_params['testing_t_window'] = [0.0, max_time]
    mob.simulate(max_time=max_time, lazy_contacts=True)

    header.append(
        'Target cases per age group at t=0:   {} {}'.format(sim_cases[0].sum().item(), list(sim_cases[0].tolist())))
    header.append(
        'Target cases per age group at t=T:   {} {}'.format(sim_cases[-1].sum().item(), list(sim_cases[-1].tolist())))
    header.append(
        'Daily test capacity in sim.:         {}'.format(testing_params['tests_per_batch']))

    # instantiate correct distributions
    distributions = CovidDistributions(country=args.country)

    # set Bayesian optimization target as positive cases
    n_days, n_age = sim_cases.shape
    
    sim_bounds = pdict_to_parr(
        pdict=param_bounds, 
        multi_beta_calibration=multi_beta_calibration
    ).T

    n_params = sim_bounds.shape[1]

    header.append(f'Parameters : {n_params}')
    header.append('Parameter bounds: {}'.format(parr_to_pdict(parr=sim_bounds.T, multi_beta_calibration=multi_beta_calibration)))

    # extract lockdown period
    sim_start_date = pd.to_datetime(data_start_date)
    sim_end_date = sim_start_date + timedelta(days=int(max_time / TO_HOURS))

    lockdown_start_date = pd.to_datetime(
        calibration_lockdown_dates[args.country]['start'])
    lockdown_end_date = pd.to_datetime(
        calibration_lockdown_dates[args.country]['end'])

    days_until_lockdown_start = (lockdown_start_date - sim_start_date).days
    days_until_lockdown_end = (lockdown_end_date - sim_start_date).days

    header.append(f'Simulation starts at : {sim_start_date}')
    header.append(f'             ends at : {sim_end_date}')
    header.append(f'Lockdown   starts at : {lockdown_start_date}')
    header.append(f'             ends at : {lockdown_end_date}')
    header.append(f'Cases compared until : {pd.to_datetime(data_end_date)}')
    header.append(f'            for days : {sim_cases.shape[0]}')
    
    # create settings dictionary for simulations
    launch_kwargs = dict(
        mob_settings=mob_settings,
        distributions=distributions,
        random_repeats=simulation_roll_outs,
        cpu_count=cpu_count,
        initial_seeds=initial_seeds,
        testing_params=testing_params,
        max_time=max_time,
        num_people=mob.num_people,
        num_sites=mob.num_sites,
        home_loc=mob.home_loc,
        site_loc=mob.site_loc,
        lazy_contacts=lazy_contacts,
        verbose=False)


    '''
    Define central functions for optimization
    '''

    G_obs = torch.tensor(sim_cases).reshape(1, n_days * n_age)
    G_obs_aggregate = torch.tensor(sim_cases).sum(dim=-1)

    '''
    Objective function
    Note: in BO and botorch, objectives are maximized
    '''
    if per_age_group_objective:
        def composite_squared_loss(G):
            return - (G - G_obs).pow(2).sum(dim=-1) / n_days

    else:
        def composite_squared_loss(G):
            return - (G - G_obs_aggregate).pow(2).sum(dim=-1) / n_days


    # select objective function
    objective = GenericMCObjective(composite_squared_loss)

    def case_diff(preds):
        '''
        Computes aggregate case difference of predictions and ground truth at t=T
        '''
        if per_age_group_objective:
            return preds[-1].sum(dim=-1) - G_obs_aggregate[-1]
        else:
            return preds[-1] - G_obs_aggregate[-1]

    def unnormalize_theta(theta):
        '''
        Computes unnormalized parameters
        '''
        return transforms.unnormalize(theta, sim_bounds)

    def composite_simulation(norm_params):
        """
        Takes a set of normalized (unit cube) BO parameters
        and returns simulator output means and standard errors based on multiple
        random restarts. This corresponds to the black-box function.
        """

        # un-normalize normalized params to obtain simulation parameters
        params = transforms.unnormalize(norm_params, sim_bounds)

        # finalize model parameters based on given parameters and calibration mode
        kwargs = copy.deepcopy(launch_kwargs)        
        all_params = parr_to_pdict(parr=params, multi_beta_calibration=multi_beta_calibration)

        if multi_beta_calibration:
            betas = all_params['betas']
        else:
            betas = {
                'education': all_params['beta_site'],
                'social': all_params['beta_site'],
                'bus_stop': all_params['beta_site'],
                'office': all_params['beta_site'],
                'supermarket': all_params['beta_site'],
            }

        model_params = {
            'betas' : betas,
            'beta_household' : all_params['beta_household'],
        }

        # set exposure parameters
        kwargs['params'] = model_params

        # set measure parameters
        kwargs['measure_list'] = MeasureList([
            # standard behavior of positively tested: full isolation
            SocialDistancingForPositiveMeasure(
                t_window=Interval(0.0, max_time), p_stay_home=1.0),
            SocialDistancingForPositiveMeasureHousehold(
                t_window=Interval(0.0, max_time), p_isolate=1.0),

            # social distancing factor during lockdown: calibrated
            SocialDistancingForAllMeasure(
                t_window=Interval(TO_HOURS * days_until_lockdown_start,
                                  TO_HOURS * days_until_lockdown_end),
                p_stay_home=all_params['p_stay_home']),

            # site specific measures: fixed in advance, outside of calibration
            BetaMultiplierMeasureByType(
                t_window=Interval(TO_HOURS * days_until_lockdown_start,
                                  TO_HOURS * days_until_lockdown_end),
                beta_multiplier=calibration_lockdown_beta_multipliers)
        ])

        # run simulation in parallel,
        summary = launch_parallel_simulations(**kwargs)

        # (random_repeats, n_people)
        posi_started = torch.tensor(summary.state_started_at['posi'])
        posi_started -= test_lag_days * TO_HOURS # account for test lag in objective computation

        # (random_repeats, n_days)
        age_groups = torch.tensor(summary.people_age)

        # (random_repeats, n_days, n_age_groups)
        posi_cumulative = convert_timings_to_cumulative_daily(
            timings=posi_started, age_groups=age_groups, time_horizon=n_days * TO_HOURS)

        if posi_cumulative.shape[0] <= 1:
            raise ValueError('Must run at least 2 random restarts per setting to get estimate of noise in observation.')
        
        # compute aggregate if not using objective per age-group
        if not per_age_group_objective:
            posi_cumulative = posi_cumulative.sum(dim=-1)

        # compute mean and standard error of means        
        G = torch.mean(posi_cumulative, dim=0)
        G_sem = torch.std(posi_cumulative, dim=0) / math.sqrt(posi_cumulative.shape[0])

        # make sure noise is not zero for non-degenerateness
        G_sem = torch.max(G_sem, MIN_NOISE)

        # flatten
        if per_age_group_objective:
            G = G.reshape(n_days * n_age)
            G_sem = G_sem.reshape(n_days * n_age)

        return G, G_sem

    def generate_initial_observations(n, logger, loaded_init_theta=None, loaded_init_G=None, loaded_init_G_sem=None):
        """
        Takes an integer `n` and generates `n` initial observations
        from the black box function using Sobol random parameter settings
        in the unit cube. Returns parameter setting and black box function outputs.
        If `loaded_init_theta/G/G_sem` are specified, initialization is loaded (possibly partially, in which
        case the initialization using the Sobol random sequence is continued where left off).
        """

        if n <= 0:
            raise ValueError(
                'qKnowledgeGradient and GP needs at least one observation to be defined properly.')

        # sobol sequence proposal points
        # new_thetas: [n, n_params]
        new_thetas = torch.tensor(
            sobol_seq.i4_sobol_generate(n_params, n), dtype=torch.float)

        # check whether initial observations are loaded
        loaded = (loaded_init_theta is not None
              and loaded_init_G is not None 
              and loaded_init_G_sem is not None)
        if loaded:
            n_loaded = loaded_init_theta.shape[0] # loaded no. of observations total
            n_loaded_init = min(n_loaded, n)      # loaded no. of quasi-random initialization observations
            n_init = max(n_loaded, n)             # final no. of observations returned, at least quasi-random initializations

            # check whether loaded proposal points are same as without loading observations
            try:
                assert(np.allclose(loaded_init_theta[:n_loaded_init], new_thetas[:n_loaded_init]))
            except AssertionError:
                print(
                    '\n\n\n===> Warning: parameters of loaded inital observations '
                    'do not coincide with initialization that would have been done. '
                    'Double check simulation, ninit, and parameter bounds, which could change '
                    'the initial random Sobol sequence. \nThe loaded parameter settings are used. \n\n\n'
                )
            
            if n_init > n:
                new_thetas = loaded_init_theta # size of tensor increased to `n_init`, as more than Sobol init points loaded

        else:
            n_loaded = 0       # loaded no. of observations total
            n_loaded_init = 0  # loaded no. of quasi-random initialization observations
            n_init = n         # final no. of observations returned, at least quasi-random initializations

        # instantiate simulator observation tensors
        if per_age_group_objective:
            # new_G, new_G_sem: [n_init, n_days * n_age] (flattened outputs)
            new_G = torch.zeros((n_init, n_days * n_age), dtype=torch.float)
            new_G_sem = torch.zeros((n_init, n_days * n_age), dtype=torch.float)
        else:
            # new_G, new_G_sem: [n_init, n_days]
            new_G = torch.zeros((n_init, n_days), dtype=torch.float)
            new_G_sem = torch.zeros((n_init, n_days), dtype=torch.float)

        # generate `n` initial evaluations at quasi random settings; if applicable, skip and load expensive evaluation result
        for i in range(n_init):
            
            # if loaded, use initial observation for this parameter settings
            if loaded and i <= n_loaded - 1:
                new_thetas[i] = loaded_init_theta[i]
                G, G_sem = loaded_init_G[i], loaded_init_G_sem[i]
                walltime = 0.0

            # if not loaded, evaluate as usual
            else:
                t0 = time.time()
                G, G_sem = composite_simulation(new_thetas[i])
                walltime = time.time() - t0

            new_G[i] = G
            new_G_sem[i] = G_sem

            # log
            G_objectives = objective(new_G[:i+1])
            best_idx = G_objectives.argmax()
            best = G_objectives[best_idx].item()
            current = objective(G).item()

            if per_age_group_objective:
                case_diff = G.reshape(n_days, n_age)[-1].sum() - G_obs_aggregate[-1]
            else:
                case_diff = G[-1] - G_obs_aggregate[-1]
            
            logger.log(
                i=i - n,
                time=walltime,
                best=best,
                objective=current,
                case_diff=case_diff,
                theta=transforms.unnormalize(new_thetas[i, :].detach().squeeze(), sim_bounds)
            )

            # save state
            state = {
                'train_theta': new_thetas[:i+1],
                'train_G': new_G[:i+1],
                'train_G_sem': new_G_sem[:i+1],
                'best_observed_obj': best,
                'best_observed_idx': best_idx,
            }
            save_state(state, logger.filename)

        # compute best objective from simulations
        f = objective(new_G)
        best_f_idx = f.argmax()
        best_f = f[best_f_idx].item()

        return new_thetas, new_G, new_G_sem, best_f, best_f_idx

    def initialize_model(train_x, train_y, train_y_sem):
        """
        Defines a GP given X, Y, and noise observations (standard error of mean)
        """
        
        train_ynoise = train_y_sem.pow(2.0) # noise is in variance units
        
        # standardize outputs to zero mean, unit variance to have good hyperparameter tuning
        outcome_transform = Standardize(m=n_days * n_age if per_age_group_objective else n_days)
        model = FixedNoiseGP(train_x, train_y, train_ynoise, outcome_transform=outcome_transform)

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        return mll, model

    # Model initialization
    # parameters used in BO are always in unit cube for optimal hyperparameter tuning of GPs
    bo_bounds = torch.stack([torch.zeros(n_params), torch.ones(n_params)])

    def optimize_acqf_and_get_observation(acq_func, args):
        """
        Optimizes the acquisition function, and returns a new candidate and a noisy observation.
        botorch defaults:  num_restarts=10, raw_samples=256, batch_limit=5, maxiter=200
        """

        batch_initial_conditions = gen_one_shot_kg_initial_conditions(
            acq_function=acq_func,
            bounds=bo_bounds,
            q=1,
            num_restarts=args.acqf_opt_num_restarts,
            raw_samples=args.acqf_opt_raw_samples,
            options={"batch_limit": args.acqf_opt_batch_limit,
                     "maxiter": args.acqf_opt_maxiter},
        )

        # optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bo_bounds,
            q=1,
            num_restarts=args.acqf_opt_num_restarts,
            raw_samples=args.acqf_opt_raw_samples,  # used for intialization heuristic
            options={"batch_limit": args.acqf_opt_batch_limit,
                     "maxiter": args.acqf_opt_maxiter},
            batch_initial_conditions=batch_initial_conditions
        )

        # proposed evaluation
        new_theta = candidates.detach().squeeze()

        # observe new noisy function evaluation
        new_G, new_G_sem = composite_simulation(new_theta)

        return new_theta, new_G, new_G_sem

    # return functions
    return (
        objective, 
        generate_initial_observations,
        initialize_model,
        optimize_acqf_and_get_observation,
        case_diff,
        unnormalize_theta,
        header,
    )


