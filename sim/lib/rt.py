"""
Utility function to compute effective reproduction number. Implementation is
dased on the Bayesian approach implemented in

    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb

which is based on:

    Bettencourt LMA, Ribeiro RM (2008) Real Time Bayesian Estimation of the
    Epidemic Potential of Emerging Infectious Diseases. PLOS ONE 3(5): e2185.
    https://doi.org/10.1371/journal.pone.0002185
"""
import numpy as np
import pandas as pd

from scipy import stats as sps
from scipy.optimize import minimize


# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = 1/7

# Range of discrete R_t used to evaluate the posterior distributions
R_T_MAX = 5
R_T_RANGE = np.linspace(0, R_T_MAX, R_T_MAX*10+1)


def days_to_datetime(arr, start_date):
    # timestamps
    ts = arr * 24 * 60 * 60 + pd.Timestamp(start_date).timestamp()
    return pd.to_datetime(ts, unit='s')


def prepare_cases(new_cases, window=7, cutoff=None):
    original = new_cases.copy()
    smoothed = new_cases.rolling(window,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    if cutoff is not None:
        idx_start = np.searchsorted(smoothed, cutoff)
        smoothed = smoothed.iloc[idx_start:]
        original = original.loc[smoothed.index]
    return original, smoothed


def format_simulation(sim, start_date, window=3):
    days_start_time = np.arange(0.0, sim.max_time, step=24.0)
    days_index = days_to_datetime(days_start_time/24, start_date=start_date)
    data = list()
    for r in range(sim.random_repeats):
        new_cases_r = np.zeros_like(days_start_time)
        for i, t0 in enumerate(days_start_time):
            # end of day
            t1 = t0 + 24.0
            # people that got infectious in this window
            new_cases_r[i] = (
                np.sum((sim.state_started_at['iasy'][r] >= t0) & (sim.state_started_at['iasy'][r] < t1)) +
                np.sum((sim.state_started_at['ipre'][r] >= t0) & (sim.state_started_at['ipre'][r] < t1)))
        # format as pandas series
        sr = pd.Series(new_cases_r, index=days_index)
        # Smooth out cases to get smoother Rt estimates
        original, smoothed = prepare_cases(sr, window=window, cutoff=None)
        data.append(smoothed)
    return data


def get_posteriors(sr, sigma, r_t_range):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))
    # Clip lam values to avoid all-zero Poisson densities
    lam[lam < 1e-10] = 1e-10


    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood


def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    try:
        best = (highs - lows).argmin()
        low_idx = lows[best]
        high_idx = highs[best]
    except ValueError:
        low_idx = 0
        high_idx = 0

    low = pmf.index[low_idx]
    high = pmf.index[high_idx]

    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])


def obj(sigma, data, r_t_range, verbose=False):
    # Cast sigma if necessary (for scipy.optimize.minimize)
    if isinstance(sigma, np.ndarray):
        assert len(sigma) == 1
        sigma = sigma[0]
    total_log_likelihood = 0.0
    for r, cases in enumerate(data):
        if verbose:
            print(f'{sigma:>5.2f}: {r}', end='\r', flush=True)
        posteriors, log_likelihood = get_posteriors(cases, sigma=sigma, r_t_range=r_t_range)
        total_log_likelihood += log_likelihood
    if verbose: print()
    return -total_log_likelihood


def find_sigma(data, r_t_range):
    res = minimize(fun=obj, x0=0.5, method='L-BFGS-B', bounds=[(1e-2, 1.0)],
                   args=(data, r_t_range, True))
    return res


def compute_daily_rts(sim, start_date, sigma=None, r_t_range=R_T_RANGE, window=3):
    # Format the observations
    data = format_simulation(sim, start_date, window)
    # If not provided, find the variance of the prior using MLE
    if sigma is None:
        print('Optimize sigma using maximum likelihood estimation...')
        optres = find_sigma(data, r_t_range)
        sigma = optres.x[0]
        print(f'done. Best sigma found at: {sigma:.2f}')
    # Find the R_t posteriors for all realizations (ie Monte Carlos roll-outs)
    all_posteriors = []
    for r, sr in enumerate(data):
        post_r, _ = get_posteriors(sr, sigma=sigma, r_t_range=r_t_range)
        all_posteriors.append(post_r)
    # Average posteriors over realizations
    posteriors = all_posteriors[0].copy()
    for post_r in all_posteriors[1:]:
        posteriors += post_r
    posteriors /= len(all_posteriors)
    # Aggregate high density areas of posteriors
    hdis = highest_density_interval(posteriors, p=0.9)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis], axis=1)
    return result
