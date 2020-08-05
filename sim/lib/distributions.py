
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

TO_HOURS = 24.0

class CovidDistributions(object):
    """
    Class to sample from specific distributions for SARS COV2
    """

    def __init__(self, country):

        '''
        Covid-19 specific constants from literature
        ALL UNITS IN DAYS
        '''
        self.lambda_0 = 0.0

        # https://www.medrxiv.org/content/10.1101/2020.03.03.20029983v1
        # https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf
        # https://science.sciencemag.org/content/368/6491/eabb6936/tab-pdf
        self.R0 = 2.0 # for seeding

        # Proportion of infections that are asymptomatic
        # https://www.medrxiv.org/content/10.1101/2020.02.03.20020248v2
        # https://science.sciencemag.org/content/368/6491/eabb6936/tab-pdf
        # https://www.medrxiv.org/content/10.1101/2020.04.17.20053157v1.full.pdf
        self.alpha = 0.4

        # Relative transmission rate of asymptomatic individuals
        # Li et al (Science, 2020): "multiplicative factor reducing the transmission rate of unreported infected patients"
        # https://science.sciencemag.org/content/368/6490/489/tab-pdf 
        self.mu = 0.55

        # https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0
        # https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf
        # https://www.bag.admin.ch/bag/en/home/krankheiten/ausbrueche-epidemien-pandemien/aktuelle-ausbrueche-epidemien/novel-cov/situation-schweiz-und-international.html 
        if country == 'GER':
            self.fatality_rates_by_age = np.array([0.0, 0.0, 0.0, 0.004, 0.073, 0.247])
            self.p_hospital_by_age = np.array([0.001, 0.002, 0.012, 0.065, 0.205, 0.273])
        elif country == 'CH':
            self.fatality_rates_by_age = np.array([0, 0, 0, 0.001, 0.001, 0.005, 0.031, 0.111, 0.265])
            self.p_hospital_by_age = np.array([0.155, 0.038, 0.028, 0.033, 0.054, 0.089, 0.178, 0.326, 0.29])
        else:
            raise NotImplementedError('Invalid country requested.')
        
        # https://www.medrxiv.org/content/10.1101/2020.03.09.20033217v2
        self.gamma = np.log(2.0) / 2.0 # approximately 2 hour half life
        self.delta = np.log(5.0) / self.gamma # time of intensity decrease to below 20 %
       
        # Incubation period: estimated mean is 5.52 days, std dev is 2.41 days
        # To be able to approx. represent latent and infectious incubation period separately,
        # we subtract the estimated median infectious time period from the above mean
        # when sampling the arrival times. The infectious period std dev is heuristically set to 1 day.
        
        # https://www.medrxiv.org/content/10.1101/2020.03.05.20030502v1
        # https://www.nature.com/articles/s41591-020-0869-5
        # https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf
        # https://jamanetwork.com/journals/jama/fullarticle/2761044
        # https://pubmed.ncbi.nlm.nih.gov/32079150/
    
        self.incubation_mean_of_lognormal = 5.52
        self.incubation_std_of_lognormal = 2.41
        self.median_infectious_without_symptom = 2.3

        # Other literature parameters       
        self.median_symp_to_resi = 14.0   
        self.median_asymp_to_resi = 14.0   
        self.median_symp_to_hosp = 7.0
        self.symp_to_death_mean_of_lognormal = 13.0

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
        return TO_HOURS * np.random.exponential(
            scale=1.0 / self.lambda_0, 
            size=size)

    def sample_expo_ipre(self, size=1):
        '''
        Samples r.v. of expo -> ipre 
        '''
        return TO_HOURS * (self.__mean_distribution(
            self.incubation_mean_of_lognormal - self.median_infectious_without_symptom, 
            self.incubation_std_of_lognormal, size=size))

    def sample_expo_iasy(self, size=1):
        '''
        Samples r.v. of expo -> iasy
        '''
        return TO_HOURS * (self.__mean_distribution(
            self.incubation_mean_of_lognormal - self.median_infectious_without_symptom, 
            self.incubation_std_of_lognormal, size=size))

    def sample_ipre_isym(self, size=1):
        '''
        Samples r.v. of ipre -> isym (Incubation period)
        '''
        return TO_HOURS * self.__mean_distribution(self.median_infectious_without_symptom, 1.0, size=size)

    def sample_isym_resi(self, size=1):
        '''
        Samples r.v. of isym -> resi 
        '''
        return TO_HOURS * self.__mean_distribution(self.median_symp_to_resi, 1.0, size=size)

    def sample_isym_dead(self, size=1):
        '''
        Samples r.v. of isym -> dead
        '''
        return TO_HOURS * self.__mean_distribution(self.symp_to_death_mean_of_lognormal, 1.0, size=size)

    def sample_isym_hosp(self, size=1):
        '''
        Samples r.v. of isym -> hosp
        '''
        return TO_HOURS * self.__mean_distribution(self.median_symp_to_hosp, 1.0, size=size)


    def sample_iasy_resi(self, size=1):
        '''
        Samples r.v. of iasy -> resi
        '''
        return TO_HOURS * self.__mean_distribution(self.median_asymp_to_resi, 1.0, size=size)

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

    print('isym to dead  : ',
          dist.normal_to_lognormal(dist.symp_to_death_mean_of_lognormal, 1.0))

    print('isym to hosp  : ',
          dist.normal_to_lognormal(dist.median_symp_to_hosp, 1.0))

    print('iasy to resi  : ',
          dist.normal_to_lognormal(dist.median_asymp_to_resi, 1.0))
