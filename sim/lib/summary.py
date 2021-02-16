import os
from collections import namedtuple

import numpy as np
from lib.measures import UpperBoundCasesBetaMultiplier, SocialDistancingForAllMeasure, SocialDistancingForSmartTracing, \
    SocialDistancingByAgeMeasure, SocialDistancingForPositiveMeasure, SocialDistancingForSmartTracingHousehold, \
    SocialDistancingSymptomaticAfterSmartTracing, SocialDistancingSymptomaticAfterSmartTracingHousehold
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


def load_condensed_summary_compat(summary_path):
    """ Compatibility mode version for `load_condensed_summary` handling the case of having to extract
     condensed summary from full summary"""
    try:
        data = load_condensed_summary(summary_path)
    except FileNotFoundError:
        _ = create_condensed_summary_from_path(summary_path, acc=500)
        data = load_condensed_summary(summary_path)
    return data


def condense_summary(summary, metadata=None, acc=500):
    result = Result(metadata=metadata, summary=summary)
    try:    # For compatibility reasons
        n_age_groups = metadata.num_age_groups
    except (KeyError, AttributeError):
        n_age_groups = None

    if acc > summary.max_time:
        acc = int(summary.max_time)
        print(f'Requested accuracy not attainable, using maximal acc={acc} ...')

    ts, iasy_mu, iasy_sig = comp_state_over_time(summary, 'iasy', acc)
    _, ipre_mu, ipre_sig = comp_state_over_time(summary, 'ipre', acc)
    _, isym_mu, isym_sig = comp_state_over_time(summary, 'isym', acc)
    _, hosp_mu, hosp_sig = comp_state_over_time(summary, 'hosp', acc)
    _, dead_mu, dead_sig = comp_state_over_time(summary, 'dead', acc)
    _, resi_mu, resi_sig = comp_state_over_time(summary, 'resi', acc)
    _, posi_mu, posi_sig = comp_state_over_time(summary, 'posi', acc)
    _, nega_mu, nega_sig = comp_state_over_time(summary, 'nega', acc)
    _, iasy, _ = comp_state_over_time(summary, 'iasy', acc, return_single_runs=True)
    _, ipre, _ = comp_state_over_time(summary, 'ipre', acc, return_single_runs=True)
    _, isym, _ = comp_state_over_time(summary, 'isym', acc, return_single_runs=True)

    # Daily new infections/hospitalizations/deaths
    _, new_infected_mu, new_infected_sig = comp_daily_new(summary, states=['iasy', 'isym'])
    _, new_hosp_mu, new_hosp_sig = comp_daily_new(summary, states=['hosp'])
    _, new_dead_mu, new_dead_sig = comp_daily_new(summary, states=['dead'])

    # Cumulative statistics
    _, cumu_infected_mu, cumu_infected_sig = comp_state_cumulative(summary, state=['iasy', 'isym'], acc=acc)
    _, cumu_hosp_mu, cumu_hosp_sig = comp_state_cumulative(summary, state=['hosp'], acc=acc)
    _, cumu_dead_mu, cumu_dead_sig = comp_state_cumulative(summary, state=['dead'], acc=acc)

    # Tracing/containment statistics
    # print('Start computing containment')
    # _, contained_mu, contained_sig = comp_contained_over_time(summary, acc)
    # print(contained_mu)

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
    try:
        nbinom_dist = compute_nbinom_distributions(result, x_range, interval_range)
        nbinom_rts = estimate_daily_nbinom_rts(result, slider_size=24.0, window_size=24. * 7,
                                                        end_cutoff=24. * 10)
    except KeyError:
        print('Could not save secondary infection statistics due to the time window being to small.')
        nbinom_dist, nbinom_rts = None, None

    r_eff_mu, r_eff_sig, r_eff_samples = compute_effectiv_reproduction_number(summary)

    data = {'metadata': metadata,
            'acc': acc,
            'max_time': summary.max_time,
            'ts': ts,
            'iasy': iasy, 'iasy_mu': iasy_mu, 'iasy_sig': iasy_sig,
            'ipre': ipre, 'ipre_mu': ipre_mu, 'ipre_sig': ipre_sig,
            'isym': isym, 'isym_mu': isym_mu, 'isym_sig': isym_sig,
            'hosp_mu': hosp_mu, 'hosp_sig': hosp_sig,
            'dead_mu': dead_mu, 'dead_sig': dead_sig,
            'resi_mu': resi_mu, 'resi_sig': resi_sig,
            'posi_mu': posi_mu, 'posi_sig': posi_sig,
            'nega_mu': nega_mu, 'nega_sig': nega_sig,
            'new_infected_mu': new_infected_mu, 'new_infected_sig': new_infected_sig,
            'new_hosp_mu': new_hosp_mu, 'new_hosp_sig': new_hosp_sig,
            'new_dead_mu': new_dead_mu, 'new_dead_sig': new_dead_sig,
            'cumu_infected_mu': cumu_infected_mu, 'cumu_infected_sig': cumu_infected_sig,
            'cumu_hosp_mu': cumu_hosp_mu, 'cumu_hosp_sig': cumu_hosp_sig,
            'cumu_dead_mu': cumu_dead_mu, 'cumu_dead_sig': cumu_dead_sig,
            'lockdowns': lockdowns,
            'mean_lockdown_time': mean_lockdown_time,
            'posi_mu_age': posi_mu_age,
            'posi_sig_age': posi_sig_age,
            'tracing_stats': tracing_stats,
            'nbinom_dist': nbinom_dist,
            'nbinom_rts': nbinom_rts,
            'r_eff_samples': r_eff_samples,
            'r_eff_mu': r_eff_mu,
            'r_eff_sig': r_eff_sig,
            }

    return data


def compute_effectiv_reproduction_number(summary):
    if summary.max_time > 10 * 7 * TO_HOURS:
        tmax = summary.max_time - 14 * TO_HOURS
    else:
        tmax = summary.max_time

    r_eff_samples = []
    for r in range(summary.random_repeats):
        infectious_individuals_r = state_started_before(summary, r, 'isym', tmax)
        infectious_individuals_r += state_started_before(summary, r, 'iasy', tmax)
        infectious_individuals_r += state_started_before(summary, r, 'ipre', tmax)
        infectious_individuals = infectious_individuals_r > 0

        children = (summary.children_count_iasy[r] +
                    summary.children_count_ipre[r] +
                    summary.children_count_isym[r])
        valid_children = children[infectious_individuals]
        r_eff_samples.append(np.mean(valid_children))

    r_eff_mu = np.mean(r_eff_samples)
    r_eff_sig = np.std(r_eff_samples)
    print(f'R_eff = {r_eff_mu} +- {r_eff_sig}')
    print('R_eff samples: ', r_eff_samples)
    return r_eff_mu, r_eff_sig, r_eff_samples


def is_state_at(summary, r, state, t):
    if state == 'posi' or state == 'nega':
        return (summary.state_started_at[state][r] - TEST_LAG <= t) & (summary.state_ended_at[state][r] - TEST_LAG > t)
    else:
        return (summary.state_started_at[state][r] <= t) & (summary.state_ended_at[state][r] > t)


def state_started_before(summary, r, state, t):
    if state == 'posi' or state == 'nega':
        return summary.state_started_at[state][r] - TEST_LAG <= t
    else:
        return summary.state_started_at[state][r] <= t


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


def comp_daily_new(summary, states):
    ts, means, stds = [], [], []
    days = int(summary.max_time // TO_HOURS)
    old_cases = 0
    for t in range(days):
        restarts = []
        for r in range(summary.random_repeats):
            cases_at_t = 0
            for state in states:
                cases_at_t += np.sum(state_started_before(summary, r, state, TO_HOURS*t))
            new_cases = cases_at_t - old_cases
            restarts.append(new_cases)
            old_cases = cases_at_t

        ts.append(t)
        means.append(np.mean(restarts))
        stds.append(np.std(restarts))
    return np.array(ts), np.array(means), np.array(stds)


def comp_state_cumulative(summary, state, acc):
    '''
    Computes `state` variable over time [0, self.max_time] with given accuracy `acc
    '''
    ts, means, stds = [], [], []
    state = state if isinstance(state, list) else [state]
    for t in np.linspace(0.0, summary.max_time, num=acc, endpoint=True):
        restarts = np.zeros(summary.random_repeats)
        for stat in state:
            restarts += np.asarray([np.sum(state_started_before(summary, r, stat, t))
                                    for r in range(summary.random_repeats)])
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


def comp_contained_over_time(summary, acc):
    # FIXME: This requires some changes in the structure of the measures, which we should not risk making now.
    #  Can implement this at a later point properly.
    containment_measures = [SocialDistancingForSmartTracing,
                            SocialDistancingForSmartTracingHousehold,
                            SocialDistancingSymptomaticAfterSmartTracing,
                            SocialDistancingSymptomaticAfterSmartTracingHousehold]
    ts, means, stds = [], [], []
    for t in np.linspace(0.0, summary.max_time, num=acc, endpoint=True):
        num_contained = []
        for ml in summary.measure_list:
            contained_at_t = np.zeros(summary.n_people)
            for j in range(summary.n_people):
                contained_at_t[j] = False
                for measure in containment_measures:
                    print(measure)
                    if ml.is_contained(measure, t=t, j=j):
                        contained_at_t = True
            num_contained.append(np.sum(contained_at_t))
        ts.append(t)
        means.append(np.mean(num_contained))
        stds.append(np.std(num_contained))
    return np.asarray(ts), np.asarray(means), np.asarray(stds)


# def comp_contained_over_time(summary, measure, acc):
#     '''
#     Computes `state` variable over time [0, self.max_time] with given accuracy `acc
#     '''
#     ts, means, stds = [], [], []
#     for t in np.linspace(0.0, summary.max_time, num=acc, endpoint=True):
#         restarts = [np.sum(is_contained_at(summary, r, measure, t))
#             for r in range(summary.random_repeats)]
#         ts.append(t/TO_HOURS)
#         means.append(np.mean(restarts))
#         stds.append(np.std(restarts))
#     return np.array(ts), np.array(means), np.array(stds)


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


def get_plot_data(data, quantity, mode):

    if mode == 'daily':
        # line_cases = data[f'new_{quantity}_mu']
        # error_cases = data[f'new_{quantity}_sig']

        # New way
        length = len(data[f'new_{quantity}_mu'])
        cumu_cases = data[f'cumu_{quantity}_mu']
        stepsize = len(cumu_cases)//length
        daily_cases = np.zeros(length)
        for i in range(length):
            daily_cases[i] = cumu_cases[(i+1)*stepsize] - cumu_cases[i * stepsize]
        line_cases = daily_cases
        error_cases = np.zeros(len(line_cases))

    elif mode == 'cumulative':
        line_cases = data[f'cumu_{quantity}_mu']
        error_cases = data[f'cumu_{quantity}_sig']
    elif mode == 'total':
        if quantity == 'infected':
            line_cases = data['iasy_mu'] + data['ipre_mu'] + data['isym_mu']
            error_cases = np.sqrt(np.square(data['iasy_sig']) +
                                  np.square(data['ipre_sig']) +
                                  np.square(data['isym_sig']))
        else:
            line_cases = data[f'{quantity}_mu']
            error_cases = data[f'{quantity}_sig']
    elif mode == 'weekly incidence':
        # Calculate daily new cases
        length = len(data[f'new_{quantity}_mu'])
        cumu_cases = data[f'cumu_{quantity}_mu']
        stepsize = len(cumu_cases) // length
        daily_cases = np.zeros(length)
        for i in range(length):
            daily_cases[i] = cumu_cases[(i + 1) * stepsize] - cumu_cases[i * stepsize]

        # Calculate running 7 day incidence
        incidence = np.zeros(length)
        for i in range(length):
            incidence[i] = np.sum(daily_cases[max(i - 6, 0):i])
        line_cases = incidence
        error_cases = np.zeros(length)
    else:
        NotImplementedError()
    return line_cases, error_cases
