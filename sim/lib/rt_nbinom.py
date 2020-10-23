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
