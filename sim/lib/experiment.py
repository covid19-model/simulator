import sys, os
if '..' not in sys.path:
    sys.path.append('..')

import subprocess
import pickle, multiprocessing, copy
import pandas as pd
import numpy as np
from collections import namedtuple, defaultdict
import botorch.utils.transforms as transforms
import argparse
from lib.calibrationFunctions import (
    pdict_to_parr, parr_to_pdict, save_state, load_state, 
    get_calibrated_params, gen_initial_seeds, get_test_capacity, downsample_cases)
from lib.mobilitysim import MobilitySimulator
from lib.parallel import launch_parallel_simulations
from lib.distributions import CovidDistributions
from lib.data import collect_data_from_df
from lib.measures import *
from lib.calibrationSettings import (
    calibration_lockdown_dates, 
    calibration_states,
    calibration_lockdown_dates, 
    calibration_testing_params, 
    calibration_lockdown_beta_multipliers,
    calibration_mob_paths)
from lib.summary import *

TO_HOURS = 24.0
ROOT = 'summaries'

"""Tuples representing various objects concerning a simulation and experiment"""

Simulation = namedtuple('Simulation', (

    # Generic information
    'experiment_info',          # Description of the experiment that contains the simulation
    'simulation_info',          # Description of the simulation itself
    'start_date',               # Start date
    'end_date',                 # End date
    'sim_days',                 # Days of simulation
    'country',                  # Country
    'area',                     # Area
    'random_repeats',           # Random repeats of simulation

    # Mobility and measures
    'mob_settings_file',        # Mobility settings
    'full_scale',               # Whether or not simulation is done at full scale
    'measure_list',             # Measure list
    'testing_params',           # Testing params
    'store_mob',                # Indicator of whether to return and store MobilitySimulator object and measure bernoullis

    # Model
    'model_params',             # Model parameters (from calibration)
    'distributions',            # Transition distributions
    'initial_seeds',            # Simulation seeds

    ## default arguments
    'num_age_groups',           # Number of age groups
    'beacon_config',            # dictionary containing information regarding beacon implementation
    'thresholds_roc',           # threshold values for ROC curve computation

), defaults=(None, None, None))  # NOTE: `defaults` iterable is applied from back to front, i.e. just `beacon_config` and `thresholds_roc` and `num_age_groups` has a default


Plot = namedtuple('Plot', (
    'path',    # path to result file of this simulation containing pickled `Result` namedetuple
    'label',   # label of this plot on the legend
))


"""Helper functions"""

def get_properties(objs, property):
    '''Retrieves list of properties for list of namedtuples'''
    out = []
    for o in objs:
        if isinstance(o, dict):
            out.append(o[property])
        elif isinstance(o, Simulation) or isinstance(o, Plot) or isinstance(o, Result):
            out.append(getattr(o, property))
        else:
            raise ValueError(f'Unknown type of elements in `objs`. Type is {type(o).__name__}')
    return out


def options_to_str(**options):
    return '-'.join(['{}={}'.format(k, v) for k, v in options.items()])


def process_command_line(return_parser=False):
    '''Returns command line parser for experiment configuration'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True,
                        help="specify country indicator for experiment")
    parser.add_argument("--area",  required=True,
                        help="specify area indicator for experiment")
    parser.add_argument("--cpu_count", type=int, default=multiprocessing.cpu_count(),
                        help="update default number of cpus used for parallel simulation rollouts")
    parser.add_argument("--smoke_test", action="store_true",
                        help="flag to quickly finish runs to see if something breaks")

    parser.add_argument("--p_adoption", type=float,
                        help="only run experiment with a single adoption level")
    parser.add_argument("--beta_dispersion", type=float,
                        help="only run experiment with a single beta dispersion level")
    parser.add_argument("--beacon_proportion", type=float,
                        help="only run experiment with a single beacon proportion")
    parser.add_argument("--beacon_mode",
                        help="only run experiment with a single beacon mode")
    parser.add_argument("--test_lag", type=float,
                        help="only run experiment with the specified test lag")
                        
    parser.add_argument("--mobility_reduction", action="store_true",
                        help="flag to turn off mobility reduction")
    parser.add_argument("--continued", action="store_true",
                        help="skips sub-experiments for which summaries already exist")
    if return_parser:
        return parser

    args = parser.parse_args()
    country = args.country
    area = args.area

    # check calibration state
    try:
        calibration_state_strg = calibration_states[country][area]
        if not os.path.isfile(calibration_states[country][area]):
            raise FileNotFoundError
    except KeyError:
        print(f'{country}-{area} is unknown country-area combination.')
        exit(1)
    except FileNotFoundError:
        print(f'{country}-{area} calibration not found.')
        exit(1)
    return args

def get_version_tag():
    git_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode(sys.stdout.encoding) 
    return git_commit

"""Experiment class for structured experimentation with simulations"""

class Experiment(object):
    """
    Class to organize a set of experiment simulations. One experiment objects
    contains several simulations that are stored and can be analyzed collectively. 
    """

    def __init__(self, *, 
        experiment_info,
        start_date,
        end_date,
        random_repeats,
        full_scale,
        verbose,
        cpu_count=None,
        multi_beta_calibration=False,
        condensed_summary=True,
        continued_run=False):

        self.experiment_info = experiment_info
        self.start_date = start_date
        self.end_date = end_date
        self.random_repeats = random_repeats
        self.cpu_count = cpu_count if cpu_count else multiprocessing.cpu_count()
        self.full_scale = full_scale
        self.multi_beta_calibration = multi_beta_calibration
        self.condensed_summary = condensed_summary
        self.continued_run = continued_run
        self.verbose = verbose

        # list simulations of experiment
        self.sims = []

    def get_sim_path(self, sim):
        version_tag = get_version_tag()
        return sim.experiment_info + '-' + version_tag + '/' + sim.experiment_info + '-' + sim.simulation_info

    def save_condensed_summary(self, sim, summary):
        filepath = os.path.join('condensed_summaries', self.get_sim_path(sim) + '_condensed.pk')
        condensed_summary = condense_summary(summary=summary, metadata=sim)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as fp:
            pickle.dump(condensed_summary, fp)
        return

    def check_summary_existence(self, sim):
        filepath = os.path.join('condensed_summaries', self.get_sim_path(sim) + '_condensed.pk')
        return os.path.isfile(filepath)

    def save_run(self, sim, summary):
        filename = self.get_sim_path(sim) + '.pk'
        obj = Result(
            metadata=sim,
            summary=summary,
        )
        with open(os.path.join(ROOT, filename), 'wb') as fp:
            pickle.dump(obj, fp)
        return

    def add(self, *,
        simulation_info,
        country,
        area,        
        measure_list,
        full_scale=True,
        test_update=None,
        seed_summary_path=None,
        set_calibrated_params_to=None,
        set_initial_seeds_to=None,
        expected_daily_base_expo_per100k=0,
        beacon_config=None,
        thresholds_roc=None,
        estimate_mobility_reduction=False,
        store_mob=False):

        # Set time window based on experiment start and end date
        sim_days = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days
        max_time = TO_HOURS * sim_days  # in hours

         # extract lockdown period
        lockdown_start_date = pd.to_datetime(
            calibration_lockdown_dates[country]['start'])
        lockdown_end_date = pd.to_datetime(
            calibration_lockdown_dates[country]['end'])

        days_until_lockdown_start = (lockdown_start_date - pd.to_datetime(self.start_date)).days
        days_until_lockdown_end = (lockdown_end_date - pd.to_datetime(self.start_date)).days

        # Load mob settings        
        mob_settings_file = calibration_mob_paths[country][area][1 if full_scale else 0]
        with open(mob_settings_file, 'rb') as fp:
            mob_settings = pickle.load(fp)

        num_age_groups = len(mob_settings['mob_rate_per_age_per_type'])

        # Obtain COVID19 case date for country and area to estimate testing capacity and heuristic seeds if necessary
        unscaled_area_cases = collect_data_from_df(country=country, area=area, datatype='new',
                                                start_date_string=self.start_date, end_date_string=self.end_date)
        assert(len(unscaled_area_cases.shape) == 2)

        # Scale down cases based on number of people in town and region
        sim_cases = downsample_cases(unscaled_area_cases, mob_settings)

        # Instantiate correct state transition distributions (estimated from literature)
        distributions = CovidDistributions(country=country)

        # Expected base rate infections
        if expected_daily_base_expo_per100k > 0.0:

            # Scale expectation to simulation size
            num_people = len(mob_settings['home_loc'])
            lambda_base_expo_population = expected_daily_base_expo_per100k * (num_people / 100000)

            # Convert to individual base rate by dividing by population size; priority queue handles superposition
            lambda_base_expo_indiv = lambda_base_expo_population / num_people

            # Poisson process with rate lambda: interarrival times are Exponential r.v. with mean = 1 / lambda
            # Hence set rate of Expo r.v.s to 1 / (1 / lambda) = lambda
            distributions.lambda_0 = lambda_base_expo_indiv

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

        if set_initial_seeds_to is not None:
            initial_seeds = set_initial_seeds_to

        if set_calibrated_params_to is not None:
            calibrated_params = set_calibrated_params_to
        else:
            # Load calibrated model parameters for this area
            calibrated_params = get_calibrated_params(
                country=country, area=area, multi_beta_calibration=self.multi_beta_calibration,
                estimate_mobility_reduction=estimate_mobility_reduction)

        if self.multi_beta_calibration:
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
            'betas': betas,
            'beta_household': calibrated_params['beta_household'],
        }        

        # Add standard measure of positives staying isolated 
        measure_list += [
            # standard behavior of positively tested: full isolation
            SocialDistancingForPositiveMeasure(
                t_window=Interval(0.0, max_time), p_stay_home=1.0),
            SocialDistancingForPositiveMeasureHousehold(
                t_window=Interval(0.0, max_time), p_isolate=1.0),
        ]

        measure_list = MeasureList(measure_list)

        # Set testing conditions
        scaled_test_capacity = get_test_capacity(
            country, area, mob_settings, end_date_string=self.end_date)
        testing_params = copy.deepcopy(calibration_testing_params)
        testing_params['tests_per_batch'] = scaled_test_capacity
        testing_params['testing_t_window'] = [0.0, max_time]
        if test_update:
            testing_params = test_update(testing_params)

        # store simulation
        sim_kwargs = dict(
            # Generic information
            experiment_info=self.experiment_info,
            simulation_info=simulation_info,
            start_date=self.start_date,
            end_date=self.end_date,
            sim_days=sim_days,
            country=country,
            area=area,
            random_repeats=self.random_repeats,

            # Mobility and measures
            mob_settings_file=mob_settings_file,
            full_scale=full_scale,
            measure_list=measure_list,
            testing_params=testing_params,
            store_mob=store_mob,

            # Model
            model_params=model_params,
            distributions=distributions,
            initial_seeds=initial_seeds,
        )

        # Beacon 
        # fields are added here (even though defaulting to `None`) to double check backwards compatibility
        # with stored `Result` objects prior to implementing beacon functionality
        if beacon_config is not None:
            sim_kwargs['beacon_config'] = beacon_config
        if thresholds_roc is not None:
            sim_kwargs['thresholds_roc'] = thresholds_roc

        sim = Simulation(**sim_kwargs)

        if self.continued_run and self.check_summary_existence(sim):
            if self.verbose:
                print(f'[Skipped Sim] {self.get_sim_path(sim)}')
        else:
            self.sims.append(sim)
            if self.verbose:
                print(f'[Added Sim] {self.get_sim_path(self.sims[-1])}')


    def run_all(self):

        '''
        Runs all simulations that were provided via the `add` method and stored in `self.sims`
        '''

        # generate experiment folder
        current_directory = os.getcwd()
        directory = os.path.join(current_directory, ROOT, self.experiment_info + '-' + get_version_tag())
        # directory = os.path.join(current_directory, ROOT, self.get_sim_path(self.sims[0]))
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # run all simulations
        for sim in self.sims:

            with open(sim.mob_settings_file, 'rb') as fp:
                mob_settings = pickle.load(fp)
        
            summary = launch_parallel_simulations(
                mob_settings=sim.mob_settings_file,
                distributions=sim.distributions,
                random_repeats=sim.random_repeats,
                cpu_count=self.cpu_count,
                params=sim.model_params,
                initial_seeds=sim.initial_seeds,
                testing_params=sim.testing_params,
                measure_list=sim.measure_list,
                max_time=TO_HOURS * sim.sim_days,
                home_loc=mob_settings['home_loc'],
                num_people=len(mob_settings['home_loc']),
                site_loc=mob_settings['site_loc'],
                num_sites=len(mob_settings['site_loc']),
                beacon_config=sim.beacon_config,
                thresholds_roc=sim.thresholds_roc if sim.thresholds_roc is not None else [],  # convert to [] if None
                store_mob=sim.store_mob,
                store_measure_bernoullis=sim.store_mob,
                verbose=False)

            if self.condensed_summary is True:
                self.save_condensed_summary(sim, summary)
            else:
                self.save_run(sim, summary)

            if self.verbose:
                print(f'[Finished Sim] {self.get_sim_path(sim)}')
 
