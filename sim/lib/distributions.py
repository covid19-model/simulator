
import time
import bisect
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
import scipy as sp
import random as rd
import os
import math
import matplotlib
import matplotlib.pyplot as plt


class CovidDistributions(object):
    """
    Class to sample from specific distributions for SARS COV2
    """

    def __init__(self, country):

        self.tadj = 24.0 

        '''
        Covid-19 specific constants from literature
        ALL UNITS IN DAYS
        '''
        self.R0 = 2.0 # for seeding

        # proportion of infections that are asymptomatic
        self.alpha = 0.4

        # Li et al (Science, 2020): "multiplicative factor reducing the transmission rate of unreported infected patients"
        self.mu = 0.55

        self.lambda_0 = 0.0

        if country == 'GER':
            self.fatality_rates_by_age = np.array([0.0, 0.0, 0.0, 0.004, 0.073, 0.247])
            self.p_hospital_by_age = np.array([0.001, 0.002, 0.012, 0.065, 0.205, 0.273])
        elif country == 'CH':
            # Data taken from: https://www.bag.admin.ch/bag/en/home/krankheiten/ausbrueche-epidemien-pandemien/aktuelle-ausbrueche-epidemien/novel-cov/situation-schweiz-und-international.html
            self.p_hospital_by_age = np.array([0.155, 0.038, 0.028, 0.033, 0.054, 0.089, 0.178, 0.326, 0.29])
            self.fatality_rates_by_age = np.array([0, 0, 0, 0.001, 0.001, 0.005, 0.031, 0.111, 0.265])
        else:
            raise NotImplementedError('Invalid country requested.')

        self.gamma = np.log(2.0) / 2.0 # 2 hour half life
        self.delta = np.log(5.0) / self.gamma # time of intensity decrease to below 20 %

        self.incubation_mean_of_lognormal = 5.52
        self.incubation_std_of_lognormal = 4.14 
        # literatures gives 4.24 but adjusted by std=1 in added dist
        # as we separate latent and infectious incubation period

        self.min_latent = 1.0
        self.median_infectious_without_symptom = 2.5
        self.median_symp_to_resi = 14.0   
        self.median_asymp_to_resi = 7.0   
        self.median_symp_to_hosp = 7.0
        self.minimum_latent_duration = 1.0
        self.symp_to_death_mean_of_lognormal = 15.0
        self.symp_to_death_std_of_lognormal = 1.0           

    def normal_to_lognormal(self, mu, std):
        '''
        Converts mean and std to lognormal mean and std
        '''
        phi = np.sqrt(np.square(std) + np.square(mu))
        lnmu = np.log(np.square(mu) / phi)
        lnstd = np.sqrt(np.log(np.square(phi) / np.square(mu)))
        return lnmu, lnstd

    def __mean_distribution(self, mu, std, size):
        '''
        Samples from log normal distribution with a specified mean and std dev
        '''
        lnmean, lnstd = self.normal_to_lognormal(mu, std)
        return np.random.lognormal(mean=lnmean, sigma=lnstd, size=size)

    def sample_susc_baseexpo(self, size=1):
        '''
        Samples r.v. of susc -> expo (at base rate only)
        '''
        # this is base rate exposure only
        assert(self.lambda_0 != 0.0)
        return self.tadj * np.random.exponential(
            scale=1.0 / self.lambda_0, 
            size=size)

    def sample_expo_ipre(self, size=1):
        '''
        Samples r.v. of expo -> ipre 
        '''
        return self.tadj * (self.__mean_distribution(
            self.incubation_mean_of_lognormal - self.median_infectious_without_symptom, 
            self.incubation_std_of_lognormal, size=size))

    def sample_expo_iasy(self, size=1):
        '''
        Samples r.v. of expo -> iasy
        '''
        return self.tadj * (self.__mean_distribution(
            self.incubation_mean_of_lognormal - self.median_infectious_without_symptom, 
            self.incubation_std_of_lognormal, size=size))

    def sample_ipre_isym(self, size=1):
        '''
        Samples r.v. of ipre -> isym (Incubation period)
        '''
        return self.tadj * self.__mean_distribution(self.median_infectious_without_symptom, 1.0, size=size)

    def sample_isym_resi(self, size=1):
        '''
        Samples r.v. of isym -> resi 
        '''
        return self.tadj * self.__mean_distribution(self.median_symp_to_resi, 1.0, size=size)

    def sample_isym_dead(self, size=1):
        '''
        Samples r.v. of isym -> dead
        '''
        return self.tadj * self.__mean_distribution(self.symp_to_death_mean_of_lognormal, self.symp_to_death_std_of_lognormal, size=size)

    def sample_isym_hosp(self, size=1):
        '''
        Samples r.v. of isym -> hosp
        '''
        return self.tadj * self.__mean_distribution(self.median_symp_to_hosp, 1.0, size=size)


    def sample_iasy_resi(self, size=1):
        '''
        Samples r.v. of iasy -> resi
        '''
        return self.tadj * self.__mean_distribution(self.median_asymp_to_resi, 1.0, size=size)

    def sample_is_fatal(self, ages, size=1):
        '''
        Samples iid Bernoulli r.v. of fatality based on age group 
        '''
        assert(ages.shape[0] == size[0])
        return np.random.binomial(1, self.fatality_rates_by_age[ages], size=size)

    def sample_is_hospitalized(self, ages, size=1):
        '''
        Samples iid Bernoulli r.v. of hospitalization based on age group 
        '''
        assert(ages.shape[0] == size[0])
        return np.random.binomial(1, self.p_hospital_by_age[ages], size=size)



if __name__ == '__main__':

    dist = CovidDistributions(country='GER')

    print('expo to ipre/iasy (subtracted infectious window before symptoms) : ', 
          dist.normal_to_lognormal(dist.incubation_mean_of_lognormal - dist.median_infectious_without_symptom, dist.incubation_std_of_lognormal))
    
    print('ipre to isym  : ',
          dist.normal_to_lognormal(dist.median_infectious_without_symptom, 1.0))

    print('isym to resi  : ',
          dist.normal_to_lognormal(dist.median_symp_to_resi, 1.0))

    print('isym to resi  : ',
          dist.normal_to_lognormal(dist.median_symp_to_resi, 1.0))

    print('isym to dead  : ',
          dist.normal_to_lognormal(dist.symp_to_death_mean_of_lognormal, dist.symp_to_death_std_of_lognormal))

    print('isym to hosp  : ',
          dist.normal_to_lognormal(dist.median_symp_to_hosp, 1.0))

    print('iasy to resi  : ',
          dist.normal_to_lognormal(dist.median_asymp_to_resi, 1.0))
