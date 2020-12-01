import os
from collections import namedtuple

import numpy as np
from lib.measures import UpperBoundCasesBetaMultiplier, SocialDistancingForAllMeasure, SocialDistancingForSmartTracing, \
    SocialDistancingByAgeMeasure, SocialDistancingForPositiveMeasure
import pickle
from lib.rt_nbinom import estimate_daily_nbinom_rts, compute_nbinom_distributions


TO_HOURS = 24.0
TEST_LAG = 48.0 # hours

Result = namedtuple('Result', (
    'metadata',    # metadata of summaryulation that was run, here a `summaryulation` namedtuple
    'summary',     # result summary of summaryulation
))


def save_summary(obj, path):
    '''Saves summary file'''
    with open(os.path.join('summaries', path), 'wb') as fp:
        pickle.dump(obj, fp)


def load_summary(path):
    '''Loads summary file'''
    with open(os.path.join('summaries', path), 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def load_summary_list(paths):
    '''Loads list of several summaries'''
    objs = []
    for p in paths:
        try:
            objs.append(load_summary(p))
        except FileNotFoundError:
            print(f'{p} not found.')
    return objs


def create_condensed_summary_from_path(summary_path, acc=500):
    print(f'Extracting data from summary: {summary_path}')
    result = load_summary(summary_path)
    metadata = result[0]
    summary = result[1]
    cond_summary = condense_summary(summary, metadata, acc=acc)

    filepath = os.path.join('condensed_summaries', summary_path[:-3]+f'_condensed.pk')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as fp:
        pickle.dump(cond_summary, fp)
    print(f'Data extraction successful.')
    return cond_summary['acc']


def load_condensed_summary(summary_path, acc=None):
    with open(os.path.join('condensed_summaries', summary_path[:-3]+f'_condensed.pk'), 'rb') as fp:
        data = pickle.load(fp)
    print('Loaded previously extracted data.')
    return data


def condense_summary(summary, metadata=None, acc=500):
    result = Result(metadata=metadata, summary=summary)
    try:    # For compatibility reasons
        n_age_groups = metadata.num_age_groups
    except KeyError:
        n_age_groups = None


    if acc > summary.max_time:
        acc = int(summary.max_time)
        print(f'Requested accuracy not attainable, using maximal acc={acc} ...')

    ts, iasy_mu, iasy_sig = comp_state_over_time(summary, 'iasy', acc)
    _, ipre_mu, ipre_sig = comp_state_over_time(summary, 'ipre', acc)
    _, isym_mu, isym_sig = comp_state_over_time(summary, 'isym', acc)
    _, posi_mu, posi_sig = comp_state_over_time(summary, 'posi', acc)
    _, hosp_mu, hosp_sig = comp_state_over_time(summary, 'hosp', acc)
    _, dead_mu, dead_sig = comp_state_over_time(summary, 'dead', acc)
    _, posi_mu, posi_sig = comp_state_over_time(summary, 'posi', acc)
    _, nega_mu, nega_sig = comp_state_over_time(summary, 'nega', acc)
    _, iasy, _ = comp_state_over_time(summary, 'iasy', acc, return_single_runs=True)
    _, ipre, _ = comp_state_over_time(summary, 'ipre', acc, return_single_runs=True)
    _, isym, _ = comp_state_over_time(summary, 'isym', acc, return_single_runs=True)

    # lockdowns = None
    # mean_lockdown_time = 0
    lockdowns, mean_lockdown_time = get_lockdown_times(summary)

    posi_mu_age, posi_sig_age = [], []
    if n_age_groups:
        for age in range(n_age_groups):
            _, posi_mean, posi_std = comp_state_over_time_per_age('posi', acc, age)
            posi_mu_age.append(posi_mean)
            posi_sig_age.append(posi_std)

    # Collect data for ROC plot
    try:
        tracing_stats = summary.tracing_stats
    except AttributeError:
        tracing_stats = None

    # # Collect data for nbinomial plots
    x_range = np.arange(0, 20)
    t0_range = [50 * 24.0]
    window_size = 10.0 * 24
    interval_range = [(t0, t0 + window_size) for t0 in t0_range]
    nbinom_dist = compute_nbinom_distributions(result, x_range, interval_range)
    nbinom_rts = estimate_daily_nbinom_rts(result, slider_size=24.0, window_size=24. * 7,
                                                    end_cutoff=24. * 10)

    data = {'metadata': metadata,
            'acc': acc,
            'max_time': summary.max_time,
            'ts': ts,
            'iasy': iasy, 'iasy_mu': iasy_mu, 'iasy_sig': iasy_sig,
            'ipre': ipre, 'ipre_mu': ipre_mu, 'ipre_sig': ipre_sig,
            'isym': isym, 'isym_mu': isym_mu, 'isym_sig': isym_sig,
            'hosp_mu': hosp_mu, 'hosp_sig': hosp_sig,
            'dead_mu': dead_mu, 'dead_sig': dead_sig,
            'posi_mu': posi_mu, 'posi_sig': posi_sig,
            'nega_mu': nega_mu, 'nega_sig': nega_sig,
            'lockdowns': lockdowns,
            'mean_lockdown_time': mean_lockdown_time,
            'posi_mu_age': posi_mu_age,
            'posi_sig_age': posi_sig_age,
            'tracing_stats': tracing_stats,
            'nbinom_dist': nbinom_dist,
            'nbinom_rts': nbinom_rts,
            }

    return data


def is_state_at(summary, r, state, t):
    if state == 'posi' or state == 'nega':
        return (summary.state_started_at[state][r] - TEST_LAG <= t) & (summary.state_ended_at[state][r] - TEST_LAG > t)
    else:
        return (summary.state_started_at[state][r] <= t) & (summary.state_ended_at[state][r] > t)


def state_started_before(summary, r, state, t):
    if state == 'posi' or state == 'nega':
        return (summary.state_started_at[state][r] - TEST_LAG <= t)
    else:
        return (summary.state_started_at[state][r] <= t)


def is_contained_at(summary, r, measure, t):
    contained = np.zeros(summary.n_people, dtype='bool')
    for i in range(summary.n_people):
        if measure == 'SocialDistancingForAllMeasure':
            contained[i] = summary.measure_list[r].is_contained_prob(SocialDistancingForAllMeasure, t=t, j=i)
        elif measure == 'SocialDistancingForSmartTracing':
            contained[i] = summary.measure_list[r].is_contained_prob(SocialDistancingForSmartTracing, t=t, j=i)
        elif measure == 'SocialDistancingByAgeMeasure':
            contained[i] = summary.measure_list[r].is_contained_prob(SocialDistancingByAgeMeasure, t=t, age=summary.people_age[r, i])
        elif measure == 'SocialDistancingForPositiveMeasure':
            contained[i] = summary.measure_list[r].is_contained_prob(SocialDistancingForPositiveMeasure,
                                                                 t=t, j=i,
                                                                 state_posi_started_at=summary.state_started_at['posi'][r, :],
                                                                 state_posi_ended_at=summary.state_ended_at['posi'][r, :],
                                                                 state_resi_started_at=summary.state_started_at['resi'][r, :],
                                                                 state_dead_started_at=summary.state_started_at['dead'][r, :])
        else:
            raise ValueError('Social distancing measure unknown.')
    return contained


def comp_state_cumulative(summary, state, acc):
    '''
    Computes `state` variable over time [0, self.max_time] with given accuracy `acc
    '''
    ts, means, stds = [], [], []
    for t in np.linspace(0.0, summary.max_time, num=acc, endpoint=True):
        restarts = [np.sum(state_started_before(summary, r, state, t))
            for r in range(summary.random_repeats)]
        ts.append(t/TO_HOURS)
        means.append(np.mean(restarts))
        stds.append(np.std(restarts))
    return np.array(ts), np.array(means), np.array(stds)


def comp_state_over_time(summary, state, acc, return_single_runs=False):
    '''
    Computes `state` variable over time [0, self.max_time] with given accuracy `acc
    '''
    ts, means, stds = [], [], []
    for t in np.linspace(0.0, summary.max_time, num=acc, endpoint=True):
        restarts = [np.sum(is_state_at(summary, r, state, t))
            for r in range(summary.random_repeats)]
        if not return_single_runs:
            ts.append(t/TO_HOURS)
            means.append(np.mean(restarts))
            stds.append(np.std(restarts))
        else:
            ts.append(t/TO_HOURS)
            means.append(restarts)
            stds.append(restarts)
    return np.array(ts), np.array(means), np.array(stds)


def comp_state_over_time_per_age(summary, state, acc, age):
    '''
    Computes `state` variable over time [0, self.max_time] with given accuracy `acc
    for a given age group `age`
    '''
    ts, means, stds = [], [], []
    for t in np.linspace(0.0, summary.max_time, num=acc, endpoint=True):
        restarts = [np.sum(is_state_at(summary, r, state, t) & (summary.people_age[r] == age))
                    for r in range(summary.random_repeats)]
        ts.append(t/TO_HOURS)
        means.append(np.mean(restarts))
        stds.append(np.std(restarts))
    return np.array(ts), np.array(means), np.array(stds)


def comp_contained_over_time(summary, measure, acc):
    '''
    Computes `state` variable over time [0, self.max_time] with given accuracy `acc
    '''
    ts, means, stds = [], [], []
    for t in np.linspace(0.0, summary.max_time, num=acc, endpoint=True):
        restarts = [np.sum(is_contained_at(summary, r, measure, t))
            for r in range(summary.random_repeats)]
        ts.append(t/TO_HOURS)
        means.append(np.mean(restarts))
        stds.append(np.std(restarts))
    return np.array(ts), np.array(means), np.array(stds)


def get_lockdown_times(summary):
    interventions = []
    for ml in summary.measure_list:
        hist, t = None, 1
        while hist is None:
            # Search for active measure if conditional measure was not active initially
            try:
                hist = list(ml.find(UpperBoundCasesBetaMultiplier, t=t).intervention_history)
                t += TO_HOURS
            except AttributeError:
                hist = []
        try:
            lockdowns = [hist[0][:2]]
        except IndexError:
            lockdowns = None
        j = 0
        for k in range(len(hist)):
            if k > j:
                # If the time between two lock down periods is less than 2 days we count it as one lockdown\n",
                if hist[k][0] - lockdowns[j][1] < 2 * TO_HOURS:
                    lockdowns[j] = (lockdowns[j][0], hist[k][1])
                else:
                    lockdowns.append(hist[k][0:2])
                    j += 1
        interventions.append(lockdowns)

    lockdown_times = []
    for run in interventions:
        lockdown_time = 0
        if run is not None:
            for lockdown in run:
                if lockdown is not None:
                    lockdown_time += lockdown[1] - lockdown[0]
            lockdown_times.append(lockdown_time)
    mean_lockdown_time = np.mean(lockdown_times)
    return interventions, mean_lockdown_time