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
from collections import defaultdict

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
    # Get the indices of new infections (both asymptomatic and symptomatic)
    new_inf_indices = (
        ((sim.state_started_at['iasy'][r] >= t0) & (sim.state_started_at['iasy'][r] < t1)) |
        ((sim.state_started_at['ipre'][r] >= t0) & (sim.state_started_at['ipre'][r] < t1)))
    # Count the number of secondary cases (while in all possible states)
    num_children = (sim.children_count_ipre[r, new_inf_indices] +
                sim.children_count_isym[r, new_inf_indices] +
                sim.children_count_iasy[r, new_inf_indices])
    return num_children


def estimate_daily_secondary_infection_nbinom_dists(result, x_range, slider_size=14 * 24.0, window_size=24.0 * 7, end_cutoff=0.0):
    """
    Estimates Negative Binomial distribution parameters for number of secondary infections caused by
    infectious individuals in a sequence of time windows of length `window_size`, every `slider_size` units of time.

    Default: every 14 days, for a 7-day window, until the end of the simulation

    `x_range` indicates the finite range buckets of the NBin count data
    """
    
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
        print(f"\rEstimating secondary NBin r={r+1:2>d}/{len(rand_rep_range)}, interval=[{t0:>6.2f}, {t1:>6.2f}]...", end='')
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

    # Compute NB parameters
    df['param_n'] = df['kt']
    df['param_p'] = df['kt'] / (df['kt'] + df['Rt'])
    # Computer NB PMF
    df['nbinom_pmf'] = df.apply(lambda row: sps.nbinom.pmf(x_range, n=row['param_n'], p=row['param_p']), axis=1)
    
    return df


def estimate_daily_visit_infection_nbinom_dists(result, x_range):
    """
    Estimates Negative Binomial distribution parameters for number of infections occurring in a single visit
    by infectious individuals. For computational reasons, the window size etc. (as in estimate_daily_secondary_infection_nbinom_dists) 
    are computed in `dynamics.py` and computed online.
    
    `x_range` indicates the finite range buckets of the NBin count data
    """
    
    # Extract summary from result
    visit_expo_counts = result.summary.visit_expo_counts
    
    # Run the estimation
    res_data = []
    rand_rep_range = list()
    for r in range(result.metadata.random_repeats): 
        for (t0, t1), data in visit_expo_counts[r].items():
            print(f"\rEstimating visit NBin r={r+1:2>d}/{len(rand_rep_range)}, interval=[{t0:>6.2f}, {t1:>6.2f}]...", end='')
            fitter = NegativeBinomialFitter()
            fitter.fit(data)
            res_data.append({'r': r, 't0': t0, 't1': t1,
                        'Rt': fitter.r_, 'kt': fitter.k_,
                        'visit_expo_counts': data})
    print()

    # Format the results
    df = pd.DataFrame(res_data)

    # Ignore simulations with not enough data for fitting
    df['len_data'] = df['visit_expo_counts'].apply(len)
    df['sum_data'] = df['visit_expo_counts'].apply(sum)
    df.loc[(df['len_data'] < 10) + (df['sum_data'] < 10),'kt'] = np.nan
    df.loc[(df['len_data'] == 0),'Rt'] = 0.0  # if no cases observed

    # Compute NB parameters
    df['param_n'] = df['kt']
    df['param_p'] = df['kt'] / (df['kt'] + df['Rt'])
    # Computer NB PMF
    df['nbinom_pmf'] = df.apply(lambda row: sps.nbinom.pmf(x_range, n=row['param_n'], p=row['param_p']), axis=1)
    
    return df



def overdispersion_test(df, count_data_str, chi2_max_count=10):
    """
    Tests for overdisperison (var > mean) in count data

    Overview:
    https://www.jstor.org/stable/pdf/2681062.pdf?refreqid=excelsior%3Ad4e9e29ea50a0054d892f92a6171cfc2
    https://www.jstor.org/stable/pdf/25051417.pdf?refreqid=excelsior%3Aed5d7ced6c9071cd3737ddbfac3ed121

    """
    counts = df[['r', 't0', 't1', count_data_str]]

    # aggregate count data over all random restarts for statistical test
    counts_agg_over_restarts = defaultdict(list)
    for i, row in counts.iterrows():
        print(f'{i+1:7d} / {len(counts)}', end='\r')
        t0, t1 = row['t0'], row['t1']
        counts_agg_over_restarts[(t0, t1)] += row[count_data_str].tolist()
    print()
    
    # perform tests
    stats = []
    for i, ((t0, t1), c_list) in enumerate(counts_agg_over_restarts.items()):

        print(f'{i+1:7d} / {len(counts_agg_over_restarts)}', end='\r')

        c = np.array(c_list)
        num_obs = len(c)

        # basics 
        mean = np.mean(c)
        var = np.var(c)
        maxx = np.max(c)

        # chi squared goodness of fit test
        # aggregate counts higher than max count
        c_ceil = np.where(c > chi2_max_count, chi2_max_count, c)
        observed_freq = np.bincount(c_ceil, minlength=chi2_max_count+1).astype(np.float64)
        expected_freq = (sps.poisson.pmf(np.arange(chi2_max_count+1), mean) * num_obs).astype(np.float64)

        # chi2_pval = sps.distributions.chi2.sf(np.sum(((observed_freq - expected_freq) ** 2) / expected_freq), num_obs - 2)
        chi2_pval = sps.chisquare(f_obs=observed_freq, f_exp=expected_freq, ddof=1, axis=0)[1]

        # variance test / Poisson dispersion test
        vt = (num_obs - 1) * var / mean
        vt_pval = sps.distributions.chi2.sf(vt.astype(np.float64), num_obs - 1)

        stats.append({
            't0': t0, 't1': t1,
            'mean': mean.item(), 'var': var.item(), 'max': maxx.item(),
            'chi2_pval': chi2_pval.item(),
            'vt_pval': vt_pval.item(),
        })

    stats_df = pd.DataFrame(stats)
    return stats_df


