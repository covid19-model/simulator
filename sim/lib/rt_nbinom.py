"""
This module provides functions and classes to estimate the Effective
Reproduction Number and Dispersion by fitting a Negative-Binomial distribution
for the numbers of secondary cases, as described in:

Benjamin M. Althouse et al (2020). Stochasticity and heterogeneity in the
ransmission dynamics of SARS-CoV-2. https://arxiv.org/abs/2005.13689

Siva Athreya at al (2020). Effective Reproduction Number and Dispersion under
Contact Tracing and Lockdown on COVID-19 in Karnataka.
https://doi.org/10.1101/2020.08.19.20178111





"""
import itertools
import pandas as pd
from scipy import stats as sps


import scipy.optimize as spo
import scipy.special as spsp
import numpy as np


class NegativeBinomialFitter:

    def __init__(self, x0=[1.5, 1.0]):
        self.x0 = x0  # Initial guess

    def nbinom_log_pmf(self, x, r, k):
        return (spsp.loggamma(k + x)
                - spsp.loggamma(k)
                - spsp.loggamma(x+1)
                + k * (np.log(k) - np.log(k + r))
                + x * (np.log(r) - np.log(k + r)))

    def log_likelihood_f(self, coeffs, x, neg=1):
        n, p = coeffs
        return neg * self.nbinom_log_pmf(x, *coeffs).sum()

    def fit(self, data):
        self.res = spo.fmin(func=self.log_likelihood_f,
                            x0=self.x0, args=(data, -1),
                            full_output=True, disp=False)
        self.r_ = self.res[0][0]
        self.k_ = self.res[0][1]


def get_sec_cases_in_window(sim, r, t0, t1):
    """
    Helper function to extract the number of secondary cases from a simulation
    summary object `sim` within interval [t0, t1) for the random repeat `r`.
    """
    # Get the infices of new infections (both asymptotic and symptotic)
    new_inf_indices = (
        ((sim.state_started_at['iasy'][r] >= t0) & (sim.state_started_at['iasy'][r] < t1)) |
        ((sim.state_started_at['ipre'][r] >= t0) & (sim.state_started_at['ipre'][r] < t1)))
    # Count the number of secondary cases (while in all possible states)
    num_children = (sim.children_count_ipre[r, new_inf_indices] +
                sim.children_count_isym[r, new_inf_indices] +
                sim.children_count_iasy[r, new_inf_indices])
    return num_children


def compute_nbinom_distributions(result, x_range, interval_range):
    # Fit all intervals for all random repeats
    rand_rep_range = range(result.metadata.random_repeats)
    res_data = []
    for r, (t0, t1) in itertools.product(rand_rep_range, interval_range):
            data = get_sec_cases_in_window(result.summary, r, t0, t1)
            fitter = NegativeBinomialFitter()
            fitter.fit(data)
            res_data.append({'r': r, 't0': t0, 't1': t1,
                            'Rt': fitter.r_, 'kt': fitter.k_,
                            'num_sec_cases': data})
    # Format in dataframe
    df = pd.DataFrame(res_data)
    # Ignore simulations with not enough data for fitting
    df['len_data'] = df['num_sec_cases'].apply(len)
    df['sum_data'] = df['num_sec_cases'].apply(sum)
    df.loc[(df['len_data'] < 5) + (df['sum_data'] < 5),'kt'] = np.nan
    df.loc[(df['len_data'] == 0),'Rt'] = 0.0  # if no cases observed
    # Compute NB parameters
    df['param_n'] = df['kt']
    df['param_p'] = df['kt'] / (df['kt'] + df['Rt'])
    # Computer NB PMF
    df['nbinom_pmf'] = df.apply(lambda row: sps.nbinom.pmf(x_range, n=row['param_n'], p=row['param_p']), axis=1)
    return df


def estimate_daily_nbinom_rts(result, slider_size, window_size, end_cutoff):
    # Extract summary from result
    sim = result.summary
    # Build the range of time interval to estimate for
    t0_range = np.arange(0.0, sim.max_time - window_size - end_cutoff, slider_size)
    t1_range = t0_range + window_size
    interval_range = list(zip(t0_range, t1_range))
    # Run the estimation
    res_data = []
    rand_rep_range = list(range(result.metadata.random_repeats))
    for r, (t0, t1) in itertools.product(rand_rep_range, interval_range):
        print(f"\rEstimating r={r+1:2>d}/{len(rand_rep_range)}, interval=[{t0:>6.2f}, {t1:>6.2f}]...", end='')
        data = get_sec_cases_in_window(sim, r, t0, t1)
        fitter = NegativeBinomialFitter()
        fitter.fit(data)
        res_data.append({'r': r, 't0': t0, 't1': t1,
                     'Rt': fitter.r_, 'kt': fitter.k_,
                     'num_sec_cases': data})
    print()
    # Format the results
    df = pd.DataFrame(res_data)
    # Ignore simulations with not enough data for fitting
    df['len_data'] = df['num_sec_cases'].apply(len)
    df['sum_data'] = df['num_sec_cases'].apply(sum)
    df.loc[(df['len_data'] < 10) + (df['sum_data'] < 10),'kt'] = np.nan
    df.loc[(df['len_data'] == 0),'Rt'] = 0.0  # if no cases observed
    return df
