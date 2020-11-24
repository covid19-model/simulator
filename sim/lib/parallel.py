
import time
import bisect
import copy
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
import scipy as sp
import os, math
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathos.multiprocessing import ProcessingPool as Pool

from lib.dynamics import DiseaseModel
from lib.priorityqueue import PriorityQueue
from lib.measures import (MeasureList, BetaMultiplierMeasureBySite,
                      UpperBoundCasesBetaMultiplier, UpperBoundCasesSocialDistancing,
                      SocialDistancingForAllMeasure, BetaMultiplierMeasureByType,
                      SocialDistancingForPositiveMeasure, SocialDistancingByAgeMeasure, SocialDistancingForSmartTracing, ComplianceForAllMeasure)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from lib.mobilitysim import MobilitySimulator

TO_HOURS = 24.0

pp_legal_states = ['susc', 'expo', 'ipre', 'isym', 'iasy', 'posi', 'nega', 'resi', 'dead', 'hosp']


class ParallelSummary(object):
    """
    Summary class of several restarts
    """

    def __init__(self, max_time, repeats, n_people, n_sites, site_loc, home_loc, thresholds_roc=[], lazy_contacts=True):  # lazy_contacts kept here for legacy reasons of old summaries

        self.max_time = max_time
        self.random_repeats = repeats
        self.n_people = n_people
        self.n_sites = n_sites
        self.site_loc = site_loc
        self.home_loc = home_loc
        self.lazy_contacts = lazy_contacts
       
        self.state = {
            'susc': np.ones((repeats, n_people), dtype='bool'),
            'expo': np.zeros((repeats, n_people), dtype='bool'),
            'ipre': np.zeros((repeats, n_people), dtype='bool'),
            'isym': np.zeros((repeats, n_people), dtype='bool'),
            'iasy': np.zeros((repeats, n_people), dtype='bool'),
            'posi': np.zeros((repeats, n_people), dtype='bool'),
            'nega': np.zeros((repeats, n_people), dtype='bool'),
            'resi': np.zeros((repeats, n_people), dtype='bool'),
            'dead': np.zeros((repeats, n_people), dtype='bool'),
            'hosp': np.zeros((repeats, n_people), dtype='bool'),
        }

        self.state_started_at = {
            'susc': - np.inf * np.ones((repeats, n_people), dtype='float'),
            'expo': np.inf * np.ones((repeats, n_people), dtype='float'),
            'ipre': np.inf * np.ones((repeats, n_people), dtype='float'),
            'isym': np.inf * np.ones((repeats, n_people), dtype='float'),
            'iasy': np.inf * np.ones((repeats, n_people), dtype='float'),
            'posi': np.inf * np.ones((repeats, n_people), dtype='float'),
            'nega': np.inf * np.ones((repeats, n_people), dtype='float'),
            'resi': np.inf * np.ones((repeats, n_people), dtype='float'),
            'dead': np.inf * np.ones((repeats, n_people), dtype='float'),
            'hosp': np.inf * np.ones((repeats, n_people), dtype='float'),
        }
        self.state_ended_at = {
            'susc': np.inf * np.ones((repeats, n_people), dtype='float'),
            'expo': np.inf * np.ones((repeats, n_people), dtype='float'),
            'ipre': np.inf * np.ones((repeats, n_people), dtype='float'),
            'isym': np.inf * np.ones((repeats, n_people), dtype='float'),
            'iasy': np.inf * np.ones((repeats, n_people), dtype='float'),
            'posi': np.inf * np.ones((repeats, n_people), dtype='float'),
            'nega': np.inf * np.ones((repeats, n_people), dtype='float'),
            'resi': np.inf * np.ones((repeats, n_people), dtype='float'),
            'dead': np.inf * np.ones((repeats, n_people), dtype='float'),
            'hosp': np.inf * np.ones((repeats, n_people), dtype='float'),
        }
        
        self.measure_list = []
        self.mob = []
        
        self.people_age = np.zeros((repeats, n_people), dtype='int')

        self.children_count_iasy = np.zeros((repeats, n_people), dtype='int')
        self.children_count_ipre = np.zeros((repeats, n_people), dtype='int')
        self.children_count_isym = np.zeros((repeats, n_people), dtype='int')

        self.tracing_stats = { thres : {
            'isolate': 
               {'tp': np.zeros(repeats, dtype='int'), 
                'fp': np.zeros(repeats, dtype='int'), 
                'tn': np.zeros(repeats, dtype='int'), 
                'fn': np.zeros(repeats, dtype='int')},
            'test': 
               {'tp': np.zeros(repeats, dtype='int'),
                'fp': np.zeros(repeats, dtype='int'), 
                'tn': np.zeros(repeats, dtype='int'), 
                'fn': np.zeros(repeats, dtype='int')},
        } for thres in thresholds_roc}


def create_ParallelSummary_from_DiseaseModel(sim, store_mob=False):

    summary = ParallelSummary(sim.max_time, 1, sim.n_people, sim.mob.num_sites, sim.mob.site_loc, sim.mob.home_loc)

    for code in pp_legal_states:
        summary.state[code][0, :] = sim.state[code]
        summary.state_started_at[code][0, :] = sim.state_started_at[code]
        summary.state_ended_at[code][0, :] = sim.state_ended_at[code]

    summary.measure_list.append(sim.measure_list)
    if store_mob:
        summary.mob.append(sim.mob)
    
    summary.people_age[0, :] = sim.mob.people_age
        
    summary.children_count_iasy[0, :] = sim.children_count_iasy
    summary.children_count_ipre[0, :] = sim.children_count_ipre
    summary.children_count_isym[0, :] = sim.children_count_isym

    for thres in sim.tracing_stats.keys():
        for action in ['isolate', 'test']:
            for stat in ['tp', 'fp', 'tn', 'fn']:
                summary.tracing_stats[thres][action][stat][0] = sim.tracing_stats[thres][action][stat]

    return summary


def pp_launch(r, kwargs, distributions, params, initial_counts, testing_params, measure_list, max_time,
              thresholds_roc, store_mob):

    mob = MobilitySimulator(**kwargs)
    mob.simulate(max_time=max_time)

    sim = DiseaseModel(mob, distributions)

    sim.launch_epidemic(
        params=params,
        initial_counts=initial_counts,
        testing_params=testing_params,
        measure_list=measure_list,
        thresholds_roc=thresholds_roc,
        verbose=False)

    result = {
        'state' : sim.state,
        'state_started_at': sim.state_started_at,
        'state_ended_at': sim.state_ended_at,
        'measure_list' : copy.deepcopy(sim.measure_list),
        'people_age' : sim.mob.people_age,
        'children_count_iasy': sim.children_count_iasy,
        'children_count_ipre': sim.children_count_ipre,
        'children_count_isym': sim.children_count_isym,
        'tracing_stats' : sim.tracing_stats,
    }
    if store_mob:
        result['mob'] = sim.mob

    return result


def launch_parallel_simulations(mob_settings, distributions, random_repeats, cpu_count, params, 
    initial_seeds, testing_params, measure_list, max_time, num_people, num_sites, site_loc, home_loc,
    beacon_config=None, thresholds_roc=None, verbose=True, synthetic=False, summary_options=None,
    store_mob=False, store_measure_bernoullis=False):

    with open(mob_settings, 'rb') as fp:
        kwargs = pickle.load(fp)

        # test-time mobility simulator additions and modifications
        kwargs['beacon_config'] = beacon_config

    mob_setting_list = [copy.deepcopy(kwargs) for _ in range(random_repeats)]
    distributions_list = [copy.deepcopy(distributions) for _ in range(random_repeats)]
    measure_list_list = [copy.deepcopy(measure_list) for _ in range(random_repeats)]
    params_list = [copy.deepcopy(params) for _ in range(random_repeats)]
    initial_seeds_list = [copy.deepcopy(initial_seeds) for _ in range(random_repeats)]
    testing_params_list = [copy.deepcopy(testing_params) for _ in range(random_repeats)]
    thresholds_roc_list = [copy.deepcopy(thresholds_roc) for _ in range(random_repeats)]
    max_time_list = [copy.deepcopy(max_time) for _ in range(random_repeats)]
    store_mob_list = [copy.deepcopy(store_mob) for _ in range(random_repeats)]
    repeat_ids = list(range(random_repeats))

    if verbose:
        print('Launching simulations...')

    with ProcessPoolExecutor(cpu_count) as ex:
        res = ex.map(pp_launch, repeat_ids, mob_setting_list, distributions_list, params_list,
                     initial_seeds_list, testing_params_list, measure_list_list, max_time_list,
                     thresholds_roc_list, store_mob_list)

    # # # DEBUG mode (to see errors printed properly)
    # res = []
    # for r in repeat_ids:
    #     res.append(pp_launch(r, mob_setting_list[r], distributions_list[r], params_list[r],
    #                  initial_seeds_list[r], testing_params_list[r], measure_list_list[r], 
    #                  max_time_list[r], thresholds_roc_list[r], store_mob_list[r]))

    
    # collect all result (the fact that mob is still available here is due to the for loop)
    summary = ParallelSummary(max_time, random_repeats, num_people, num_sites, site_loc, home_loc, thresholds_roc)
    
    for r, result in enumerate(res):

        for code in pp_legal_states:
            summary.state[code][r, :] = result['state'][code]
            summary.state_started_at[code][r, :] = result['state_started_at'][code]
            summary.state_ended_at[code][r, :] = result['state_ended_at'][code]

        ml = result['measure_list']
        if not store_measure_bernoullis:
            ml.exit_run()
        summary.measure_list.append(ml)

        if store_mob:
            summary.mob.append(result['mob']) 

        summary.people_age[r, :] = result['people_age']
        
        summary.children_count_iasy[r, :] = result['children_count_iasy']
        summary.children_count_ipre[r, :] = result['children_count_ipre']
        summary.children_count_isym[r, :] = result['children_count_isym']

        for thres in result['tracing_stats'].keys():
            for action in ['isolate', 'test']:
                for stat in ['tp', 'fp', 'tn', 'fn']:
                    summary.tracing_stats[thres][action][stat][r] = result['tracing_stats'][thres][action][stat]

    return summary
