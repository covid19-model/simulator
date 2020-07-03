
from lib.calibrationFunctions import get_calibrated_params
from lib.calibrationSettings import calibration_lockdown_dates, calibration_mob_paths, calibration_states
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.mobilitysim import MobilitySimulator
from lib.measures import *
import argparse
import multiprocessing
import pickle
import pandas as pd
import random as rd
import numpy as np
from tqdm import tqdm
from pprint import pprint
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

import os
if '..' not in sys.path:
    sys.path.append('..')


TO_HOURS = 24.0


def get_stats(mob, max_people, verbose=False):
    '''Computes contact statistics averaged over individuals in `mob`
    
    -number of unique contacts
    -average time per unique contact
    '''

    counts = []
    counts_unique = []
    total_contact_time = []
    ave_contact_time = []
    ave_contact_time_unique = []

    n_people = min(mob.num_people, max_people)

    iter = (tqdm(np.random.choice(mob.num_people, size=n_people), desc='n_contacts')
            if verbose else np.random.choice(mob.num_people, size=n_people))

    for j in iter:
        contacts_j = mob.find_contacts_of_indiv(indiv=j, tmin=0)
        
        if len(contacts_j) > 0:
            t = 0
            unique = set()
            for c in contacts_j:
                t += c.duration
                unique.add(c.indiv_i)

            counts.append(len(contacts_j))
            counts_unique.append(len(unique))
            total_contact_time.append(t)
            ave_contact_time.append(t / len(contacts_j))
            ave_contact_time_unique.append(t / len(unique))
        else:
            counts.append(0)
            counts_unique.append(0)
            total_contact_time.append(0.0)
            ave_contact_time.append(0.0)
            ave_contact_time_unique.append(0.0)

    return dict(
        ave_contact_counts=np.mean(counts),
        ave_contact_counts_unique=np.mean(counts_unique),
        ave_total_contact_time=np.mean(total_contact_time),
        ave_ave_contact_time=np.mean(ave_contact_time),
        ave_ave_contact_time_unique=np.mean(ave_contact_time_unique),
    )


def compute_mob_statistics(loc_tup, days, max_people, verbose=False):
    '''Computes all MobilitySimulator statistics for given `country` and `area` '''

    country, area = loc_tup

    if verbose:
        print(country, area)

    # get mobility simulator settings
    statistics = dict()
    mob_settings_downsampled, mob_settings_full = calibration_mob_paths[country][area]

    # downsampled
    with open(mob_settings_downsampled, 'rb') as fp:
        obj = pickle.load(fp)
    mob_downsampled = MobilitySimulator(**obj)
    mob_downsampled.verbose = verbose
    mob_downsampled.simulate(max_time=days * TO_HOURS, lazy_contacts=True)

    # full
    with open(mob_settings_full, 'rb') as fp:
        obj = pickle.load(fp)
    mob_full = MobilitySimulator(**obj)
    mob_full.verbose = verbose
    mob_full.simulate(max_time=days * TO_HOURS, lazy_contacts=True)

    # compute contact information
    contact_info_downsampled = get_stats(mob_downsampled, max_people, verbose=verbose)
    del mob_downsampled
    contact_info_full = get_stats(mob_full, max_people, verbose=verbose)
    del mob_full

    statistics['ratio-' + 'ave_contact_counts'] = (
        contact_info_downsampled['ave_contact_counts']
        / contact_info_full['ave_contact_counts']
    )
    statistics['ratio-' + 'ave_contact_counts_unique'] = (
        contact_info_downsampled['ave_contact_counts_unique']
        / contact_info_full['ave_contact_counts_unique']
    )

    statistics['ratio-' + 'ave_total_contact_time'] = (
        contact_info_downsampled['ave_total_contact_time']
        / contact_info_full['ave_total_contact_time']
    )

    statistics['ratio-' + 'ave_ave_contact_time'] = (
        contact_info_downsampled['ave_ave_contact_time']
        / contact_info_full['ave_ave_contact_time']
    )

    statistics['ratio-' + 'ave_ave_contact_time_unique'] = (
        contact_info_downsampled['ave_ave_contact_time_unique']
        / contact_info_full['ave_ave_contact_time_unique']
    )

    # print always
    print(country, area)
    pprint(statistics)

    return statistics

if __name__ == '__main__':

    days = 7.0
    max_people = 2000
    parallel = False
    cpu_count = 2

    locs = [
        ('GER', 'TU'), ('GER', 'KL'), ('GER', 'RH'), ('GER', 'TR'),
        ('CH', 'VD'), ('CH', 'BE'), ('CH', 'TI'), ('CH', 'JU'),
    ]

    # run in parallel for all locs
    if parallel:
        with ProcessPoolExecutor(cpu_count) as ex:
            res = ex.map(
                compute_mob_statistics, 
                locs, 
                [days for _ in locs],
                [max_people for _ in locs]
            )
    else:
        res = [compute_mob_statistics(tup, days, max_people, verbose=True) for tup in locs]


    # print all statistics
    all_statistics_unordered = dict(zip(locs, res))
    all_statistics = dict()
    stats = [
        'ratio-' + 'ave_contact_counts',
        'ratio-' + 'ave_contact_counts_unique',
        'ratio-' + 'ave_total_contact_time',
        'ratio-' + 'ave_ave_contact_time',
        'ratio-' + 'ave_ave_contact_time_unique',
    ]

    for s in stats:
        all_statistics[s] = dict()
        for loc_tup in locs:
            all_statistics[s][loc_tup] = all_statistics_unordered[loc_tup][s]
    
    print('\nStatistics by type:')
    pprint(all_statistics)
