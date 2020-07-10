import time
import bisect
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
from scipy.interpolate import interp1d
import scipy as sp
import random as rd
import os, math
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num, num2date

from lib.measures import (MeasureList, BetaMultiplierMeasureBySite,
                      SocialDistancingForAllMeasure, BetaMultiplierMeasureByType,
                      SocialDistancingForPositiveMeasure, SocialDistancingByAgeMeasure, SocialDistancingForSmartTracing, ComplianceForAllMeasure)
from lib.rt import compute_daily_rts, R_T_RANGE

import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

TO_HOURS = 24.0
DPI = 200
NO_PLOT = False
TEST_LAG = 48.0 # hours

matplotlib.rcParams.update({
    "figure.autolayout": False,
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.8,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.8,
    "text.usetex": True,
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "legend.frameon": True,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
})


def days_to_datetime(arr, start_date):
    # timestamps
    ts = arr * 24 * 60 * 60 + pd.Timestamp(start_date).timestamp()
    return pd.to_datetime(ts, unit='s')


def lockdown_widget(lockdown_at, start_date, lockdown_label_y, ymax,
                    lockdown_label, ax, ls='--', xshift=0.0, zorder=None, color='black'):
    # Convert x-axis into posix timestamps and use pandas to plot as dates
    lckdn_x = days_to_datetime(lockdown_at, start_date=start_date)
    ax.plot([lckdn_x, lckdn_x], [0, ymax], linewidth=2.5, linestyle=ls,
            color=color, label='_nolegend_', zorder=zorder)
    lockdown_label_y = lockdown_label_y or ymax*0.4
    ax.text(x=lckdn_x - pd.Timedelta(2.1 + xshift, unit='d'),
            y=lockdown_label_y, s=lockdown_label, rotation=90)


def target_widget(show_target,start_date, ax, zorder=None):
    txx = np.linspace(0, show_target.shape[0] - 1, num=show_target.shape[0])
    txx = days_to_datetime(txx, start_date=start_date)
    ax.plot(txx, show_target, linewidth=4, linestyle='', marker='X', ms=6,
            color='black', label='COVID-19 case data', zorder=zorder)


class Plotter(object):
    """
    Plotting class
    """

    def __init__(self):

        # plot constants
        # check out https://colorhunt.co/

        self.color_expo = '#ffcc00'
        self.color_iasy = '#00a8cc'
        self.color_ipre = '#005082'
        self.color_isym = '#000839'

        self.color_testing = '#ffa41b'

        self.color_posi = '#21bf73'
        self.color_nega = '#fd5e53'

        self.color_all = '#ffa41b'
        self.color_positive = '#00a8cc'
        self.color_age = '#005082'
        self.color_tracing = '#000839'

        self.color_infected = '#000839'

        self.filling_alpha = 0.2

        self.color_different_scenarios = [
            '#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
            '#ffff33',
            '#a65628',
            '#f781bf',
            '#999999'
        ]

        self.color_different_scenarios_alt = [
            '#a1dab4',
            '#41b6c4',
            '#2c7fb8',
            '#253494',
        ]



        # sequential
        # self.color_different_scenarios = [
        #     # '#ffffcc',
        #     '#c7e9b4',
        #     '#7fcdbb',
        #     '#41b6c4',
        #     '#2c7fb8',
        #     '#253494',
        #     '#000000'
        # ]



        # 2D visualization
        self.density_alpha = 0.7

        self.marker_home = "^"
        self.marker_site = "o"

        self.color_home = '#000839'
        self.color_site = '#000000'

        self.size_home = 80
        self.size_site = 300



    def __is_state_at(self, sim, r, state, t):
        if state == 'posi' or state == 'nega':
            return (sim.state_started_at[state][r] - TEST_LAG <= t) & (sim.state_ended_at[state][r] - TEST_LAG > t)
        else:
            return (sim.state_started_at[state][r] <= t) & (sim.state_ended_at[state][r] > t)

    def __state_started_before(self, sim, r, state, t):
        if state == 'posi' or state == 'nega':
            return (sim.state_started_at[state][r] - TEST_LAG <= t)
        else:
            return (sim.state_started_at[state][r] <= t)

    def __is_contained_at(self, sim, r, measure, t):
        contained = np.zeros(sim.n_people, dtype='bool')
        for i in range(sim.n_people):
            if measure == 'SocialDistancingForAllMeasure':
                contained[i] = sim.measure_list[r].is_contained_prob(SocialDistancingForAllMeasure, t=t, j=i)
            elif measure == 'SocialDistancingForSmartTracing':
                contained[i] = sim.measure_list[r].is_contained_prob(SocialDistancingForSmartTracing, t=t, j=i)
            elif measure == 'SocialDistancingByAgeMeasure':
                contained[i] = sim.measure_list[r].is_contained_prob(SocialDistancingByAgeMeasure, t=t, age=sim.people_age[r, i])
            elif measure == 'SocialDistancingForPositiveMeasure':
                contained[i] = sim.measure_list[r].is_contained_prob(SocialDistancingForPositiveMeasure,
                                                                     t=t, j=i,
                                                                     state_posi_started_at=sim.state_started_at['posi'][r, :],
                                                                     state_posi_ended_at=sim.state_ended_at['posi'][r, :],
                                                                     state_resi_started_at=sim.state_started_at['resi'][r, :],
                                                                     state_dead_started_at=sim.state_started_at['dead'][r, :])
            else:
                raise ValueError('Social distancing measure unknown.')
        return contained

    def __comp_state_cumulative(self, sim, state, acc):
        '''
        Computes `state` variable over time [0, self.max_time] with given accuracy `acc
        '''
        ts, means, stds = [], [], []
        for t in np.linspace(0.0, sim.max_time, num=acc, endpoint=True):
            restarts = [np.sum(self.__state_started_before(sim, r, state, t))
                for r in range(sim.random_repeats)]
            ts.append(t/TO_HOURS)
            means.append(np.mean(restarts))
            stds.append(np.std(restarts))
        return np.array(ts), np.array(means), np.array(stds)

    def __comp_state_over_time(self, sim, state, acc, return_single_runs=False):
        '''
        Computes `state` variable over time [0, self.max_time] with given accuracy `acc
        '''
        ts, means, stds = [], [], []
        for t in np.linspace(0.0, sim.max_time, num=acc, endpoint=True):
            restarts = [np.sum(self.__is_state_at(sim, r, state, t))
                for r in range(sim.random_repeats)]
            if not return_single_runs:
                ts.append(t/TO_HOURS)
                means.append(np.mean(restarts))
                stds.append(np.std(restarts))
            else:
                ts.append(t/TO_HOURS)
                means.append(restarts)
                stds.append(restarts)
        return np.array(ts), np.array(means), np.array(stds)

    def __comp_state_over_time_per_age(self, sim, state, acc, age):
        '''
        Computes `state` variable over time [0, self.max_time] with given accuracy `acc
        for a given age group `age`
        '''
        ts, means, stds = [], [], []
        for t in np.linspace(0.0, sim.max_time, num=acc, endpoint=True):
            restarts = [np.sum(self.__is_state_at(sim, r, state, t) & (sim.people_age[r] == age))
                        for r in range(sim.random_repeats)]
            ts.append(t/TO_HOURS)
            means.append(np.mean(restarts))
            stds.append(np.std(restarts))
        return np.array(ts), np.array(means), np.array(stds)

    def __comp_contained_over_time(self, sim, measure, acc):
        '''
        Computes `state` variable over time [0, self.max_time] with given accuracy `acc
        '''
        ts, means, stds = [], [], []
        for t in np.linspace(0.0, sim.max_time, num=acc, endpoint=True):
            restarts = [np.sum(self.__is_contained_at(sim, r, measure, t))
                for r in range(sim.random_repeats)]
            ts.append(t/TO_HOURS)
            means.append(np.mean(restarts))
            stds.append(np.std(restarts))
        return np.array(ts), np.array(means), np.array(stds)

    def plot_cumulative_infected(self, sim, title='Example', filename='daily_inf_0',
                                 figsize=(6, 5), errorevery=20, acc=1000, ymax=None,
                                 lockdown_label='Lockdown', lockdown_at=None,
                                 lockdown_label_y=None, show_target=None,
                                 start_date='1970-01-01',
                                 subplot_adjust=None, legend_loc='upper right'):
        ''''
        Plots daily infected split by group
        averaged over random restarts, using error bars for std-dev
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ts, iasy_mu, iasy_sig = self.__comp_state_cumulative(sim, 'iasy', acc)
        # _,  ipre_mu, ipre_sig = self.__comp_state_cumulative(sim, 'ipre', acc)
        _,  isym_mu, isym_sig = self.__comp_state_cumulative(sim, 'isym', acc)
        # _,  expo_mu, iexpo_sig = self.__comp_state_cumulative(sim, 'expo', acc)
        # _,  posi_mu, posi_sig = self.__comp_state_cumulative(sim, 'posi', acc)

        line_xaxis = np.zeros(ts.shape)
        line_iasy = iasy_mu
        line_isym = iasy_mu + isym_mu

        error_isym = np.sqrt(iasy_sig**2 + isym_sig**2)

        # Convert x-axis into posix timestamps and use pandas to plot as dates
        ts = days_to_datetime(ts, start_date=start_date)

        # lines
        ax.plot(ts, line_iasy, c='black', linestyle='-')
        ax.errorbar(ts, line_isym, yerr=error_isym, c='black', linestyle='-',
                    elinewidth=0.8, errorevery=errorevery, capsize=3.0)

        # filling
        ax.fill_between(ts, line_xaxis, line_iasy, alpha=self.filling_alpha, label='Asymptomatic',
                        edgecolor=self.color_iasy, facecolor=self.color_iasy, linewidth=0, zorder=0)
        ax.fill_between(ts, line_iasy, line_isym, alpha=self.filling_alpha, label='Symptomatic',
                        edgecolor=self.color_isym, facecolor=self.color_isym, linewidth=0, zorder=0)

        # limits
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')

        # extra
        if lockdown_at is not None:
            lockdown_widget(lockdown_at, start_date,
                            lockdown_label_y, ymax,
                            lockdown_label, ax)
        if show_target is not None:
            target_widget(show_target, start_date, ax)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #set ticks every week
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        ax.legend(loc=legend_loc, borderaxespad=0.5)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.draw()

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_daily_infected(self, sim, title='Example', filename='daily_inf_0',
                            figsize=(6, 5), errorevery=20, acc=1000, ymax=None,
                            lockdown_label='Lockdown', lockdown_at=None,
                            lockdown_label_y=None, show_target=None,
                            lockdown_end=None,
                            start_date='1970-01-01',
                            subplot_adjust=None, legend_loc='upper right'):
        ''''
        Plots daily infected split by group
        averaged over random restarts, using error bars for std-dev
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ts, iasy_mu, iasy_sig = self.__comp_state_over_time(sim, 'iasy', acc)
        _,  ipre_mu, ipre_sig = self.__comp_state_over_time(sim, 'ipre', acc)
        _,  isym_mu, isym_sig = self.__comp_state_over_time(sim, 'isym', acc)
        # _,  expo_mu, iexpo_sig = self.__comp_state_over_time(sim, 'expo', acc)
        # _,  posi_mu, posi_sig = self.__comp_state_over_time(sim, 'posi', acc)

        line_xaxis = np.zeros(ts.shape)
        line_iasy = iasy_mu
        line_ipre = iasy_mu + ipre_mu
        line_isym = iasy_mu + ipre_mu + isym_mu
        error_isym = np.sqrt(iasy_sig**2 + ipre_sig**2 + isym_sig**2)

        # Convert x-axis into posix timestamps and use pandas to plot as dates
        ts = days_to_datetime(ts, start_date=start_date)

        # lines
        ax.plot(ts, line_iasy,
                c='black', linestyle='-')
        ax.plot(ts, line_ipre,
                c='black', linestyle='-')
        ax.errorbar(ts, line_isym, yerr=error_isym, c='black', linestyle='-',
                    elinewidth=0.8, errorevery=errorevery, capsize=3.0)

        # filling
        ax.fill_between(ts, line_xaxis, line_iasy, alpha=0.5, label='Asymptomatic',
                        edgecolor=self.color_iasy, facecolor=self.color_iasy, linewidth=0, zorder=0)
        ax.fill_between(ts, line_iasy, line_ipre, alpha=0.5, label='Pre-symptomatic',
                        edgecolor=self.color_ipre, facecolor=self.color_ipre, linewidth=0, zorder=0)
        ax.fill_between(ts, line_ipre, line_isym, alpha=0.5, label='Symptomatic',
                        edgecolor=self.color_isym, facecolor=self.color_isym, linewidth=0, zorder=0)

        # limits
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + ipre_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')

        # extra
        if lockdown_at is not None:
            lockdown_widget(lockdown_at, start_date,
                            lockdown_label_y, ymax,
                            lockdown_label, ax)
        if lockdown_end is not None:
            lockdown_widget(lockdown_at=lockdown_end, start_date=start_date,
                            lockdown_label_y=lockdown_label_y, ymax=ymax,
                            lockdown_label='End of lockdown', ax=ax, ls='dotted')
        if show_target is not None:
            target_widget(show_target, start_date, ax)


        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #set ticks every week
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        ax.legend(loc=legend_loc, borderaxespad=0.5)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.draw()

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_daily_tested(self, sim, title='Example', filename='daily_tested_0', figsize=(10, 10), errorevery=20,
        acc=1000, ymax=None):

        ''''
        Plots daily tested, positive daily tested, negative daily tested
        averaged over random restarts, using error bars for std-dev
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # automatically shifted by `test_lag` in the function
        ts, posi_mu, posi_sig = self.__comp_state_over_time(sim, 'posi', acc)
        _,  nega_mu, nega_sig = self.__comp_state_over_time(sim, 'nega', acc)

        line_xaxis = np.zeros(ts.shape)
        line_posi = posi_mu
        line_nega = posi_mu + nega_mu

        error_posi = posi_sig
        error_nega = nega_sig + posi_sig

        T = posi_mu.shape[0]

        # lines
        ax.errorbar(ts, line_posi, yerr=posi_sig, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')
        ax.errorbar(ts, line_nega, yerr=nega_sig, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')

        # filling
        ax.fill_between(ts, line_xaxis, line_posi, alpha=self.filling_alpha, label=r'Positive tests',
                        edgecolor=self.color_posi, facecolor=self.color_posi, linewidth=0, zorder=0)
        ax.fill_between(ts, line_posi, line_nega, alpha=self.filling_alpha, label=r'Negative tests',
                        edgecolor=self.color_nega, facecolor=self.color_nega, linewidth=0, zorder=0)
        # axis
        ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(posi_mu + nega_mu)
        ax.set_ylim((0, ymax))

        ax.set_xlabel(r'$t$ [days]')
        ax.set_ylabel(r'[people]')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.70)
        ax.set_title(title, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def plot_daily_at_home(self, sim, title='Example', filename='daily_at_home_0', figsize=(10, 10), errorevery=20, acc=1000, ymax=None):

        ''''
        Plots daily tested, positive daily tested, negative daily tested
        averaged over random restarts, using error bars for std-dev
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ts, all_mu, all_sig = self.__comp_contained_over_time(sim, 'SocialDistancingForAllMeasure', acc)
        _,  positive_mu, positive_sig = self.__comp_contained_over_time(sim, 'SocialDistancingForPositiveMeasure', acc)
        _,  age_mu, age_sig = self.__comp_contained_over_time(sim, 'SocialDistancingByAgeMeasure', acc)
        _,  tracing_mu, tracing_sig = self.__comp_contained_over_time(sim, 'SocialDistancingForSmartTracing', acc)

        _, iasy_mu, iasy_sig = self.__comp_state_over_time(sim, 'iasy', acc)
        _,  ipre_mu, ipre_sig = self.__comp_state_over_time(sim, 'ipre', acc)
        _,  isym_mu, isym_sig = self.__comp_state_over_time(sim, 'isym', acc)

        line_xaxis = np.zeros(ts.shape)

        line_all = all_mu
        line_positive = positive_mu
        line_age = age_mu
        line_tracing = tracing_mu

        line_infected = iasy_mu + ipre_mu + isym_mu

        error_all = all_sig
        error_positive = positive_sig
        error_age = age_sig
        error_tracing = tracing_sig

        error_infected = np.sqrt(np.square(iasy_sig) + np.square(ipre_sig) + np.square(isym_sig))

        # lines
        ax.errorbar(ts, line_infected, label=r'Total infected', errorevery=errorevery, c=self.color_infected, linestyle='--', yerr=error_infected)

        ax.errorbar(ts, line_all, yerr=error_all, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')
        ax.errorbar(ts, line_positive, yerr=error_positive, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')
        ax.errorbar(ts, line_age, yerr=error_age, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')
        ax.errorbar(ts, line_tracing, yerr=error_tracing, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')

        # filling
        ax.fill_between(ts, line_xaxis, line_all, alpha=self.filling_alpha, label=r'SD for all',
                        edgecolor=self.color_all, facecolor=self.color_all, linewidth=0, zorder=0)
        ax.fill_between(ts, line_xaxis, line_positive, alpha=self.filling_alpha, label=r'SD for positively tested',
                        edgecolor=self.color_positive, facecolor=self.color_positive, linewidth=0, zorder=0)
        ax.fill_between(ts, line_xaxis, line_age, alpha=self.filling_alpha, label=r'SD for age group',
                        edgecolor=self.color_age, facecolor=self.color_age, linewidth=0, zorder=0)
        ax.fill_between(ts, line_xaxis, line_tracing, alpha=self.filling_alpha, label=r'SD for traced contacts',
                        edgecolor=self.color_tracing, facecolor=self.color_tracing, linewidth=0, zorder=0)

        # axis
        ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max([all_mu, positive_mu, age_mu, tracing_mu])
        ax.set_ylim((0, ymax))

        ax.set_xlabel(r'$t$ [days]')
        ax.set_ylabel(r'[people]')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.70)
        ax.set_title(title, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def compare_total_infections(self, sims, titles, figtitle='Title',
        filename='compare_inf_0', figsize=(10, 10), errorevery=20, acc=1000, ymax=None,
        lockdown_label='Lockdown', lockdown_at=None, lockdown_label_y=None,
        show_positives=False, show_legend=True, legendYoffset=0.0, legend_is_left=False, legendXoffset=0.0,
        subplot_adjust=None, start_date='1970-01-01', first_one_dashed=False, show_single_runs=False):

        ''''
        Plots total infections for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for i in range(len(sims)):
            if acc > sims[i].max_time:
                acc = int(sims[i].max_time)

            if not show_single_runs:

                ts, iasy_mu, iasy_sig = self.__comp_state_over_time(sims[i], 'iasy', acc)
                _,  ipre_mu, ipre_sig = self.__comp_state_over_time(sims[i], 'ipre', acc)
                _,  isym_mu, isym_sig = self.__comp_state_over_time(sims[i], 'isym', acc)
                _,  posi_mu, posi_sig = self.__comp_state_over_time(sims[i], 'posi', acc)

                # Convert x-axis into posix timestamps and use pandas to plot as dates
                ts = days_to_datetime(ts, start_date=start_date)

                line_xaxis = np.zeros(ts.shape)
                line_infected = iasy_mu + ipre_mu + isym_mu
                error_infected = np.sqrt(np.square(iasy_sig) + np.square(ipre_sig) + np.square(isym_sig))

                # lines
                ax.plot(ts, line_infected, linestyle='-', label=titles[i], c=self.color_different_scenarios[i])
                ax.fill_between(ts, np.maximum(line_infected - 2 * error_infected, 0), line_infected + 2 * error_infected,
                                color=self.color_different_scenarios[i], alpha=self.filling_alpha, linewidth=0.0)
            
            else:

                ts, iasy, iasy_sig = self.__comp_state_over_time(sims[i], 'iasy', acc, return_single_runs=True)
                _,  ipre, ipre_sig = self.__comp_state_over_time(sims[i], 'ipre', acc, return_single_runs=True)
                _,  isym, isym_sig = self.__comp_state_over_time(sims[i], 'isym', acc, return_single_runs=True)

                # Convert x-axis into posix timestamps and use pandas to plot as dates
                ts = days_to_datetime(ts, start_date=start_date)

                line_xaxis = np.zeros(ts.shape)
                lines_infected = iasy + ipre + isym

                # lines
                for r in range(min(show_single_runs, sims[i].random_repeats)):
                    ax.plot(ts, lines_infected[:, r], linestyle='-', label=titles[i] if r == 0 else None, 
                            c=self.color_different_scenarios[i], lw=1, alpha=0.8)



        # axis
        # ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + ipre_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')

        if not isinstance(lockdown_at, dict):
            if lockdown_at is not None:
                lockdown_widget(lockdown_at, start_date,
                                lockdown_label_y, ymax,
                                lockdown_label, ax, xshift=0.5)
        else:
            # This is only for plotting the activity of the case dependent measures (sim-conditional-measures.ipynb)
            ii = 1
            for i in lockdown_at.keys():
                interventions = lockdown_at[i]
                labels = lockdown_label[i]
                label_pos_y = lockdown_label_y[i]
                simcolor = self.color_different_scenarios[ii]
                ii += 1
                for k in range(len(interventions)):
                    lockdown_widget(interventions[k], start_date,
                                    label_pos_y, ymax,
                                    labels[k], ax, xshift=0.5, color=simcolor)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # fig.autofmt_xdate()
        #set ticks every week
        # ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        if show_legend:
            # legend
            if legend_is_left:
                leg = ax.legend(loc='upper left', borderaxespad=0.5)
            else:
                leg = ax.legend(loc='upper right', borderaxespad=0.5)

            if legendYoffset != 0.0:
                # Get the bounding box of the original legend
                bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

                # Change to location of the legend.
                bb.y0 += legendYoffset
                bb.y1 += legendYoffset
                leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
            
            if legendXoffset != 0.0:
                # Get the bounding box of the original legend
                bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

                # Change to location of the legend.
                bb.x0 += legendXoffset
                bb.x1 += legendXoffset
                leg.set_bbox_to_anchor(bb, transform=ax.transAxes)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def compare_total_fatalities_and_hospitalizations(self, sims, titles, figtitle=r'Hospitalizations and Fatalities',
        lockdown_label='Lockdown', lockdown_at=None, lockdown_label_y=None,
        filename='compare_inf_0', figsize=(10, 10), errorevery=20, acc=1000, ymax=None, 
        show_legend=True, legendYoffset=0.0, legend_is_left=False, legendXoffset=0.0,
        subplot_adjust=None, start_date='1970-01-01', first_one_dashed=False):

        ''''
        Plots total fatalities and hospitalizations for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # hospitalizations
        for i in range(len(sims)):
            if acc > sims[i].max_time:
                acc = int(sims[i].max_time)

            ts, hosp_mu, hosp_sig = self.__comp_state_over_time(
                sims[i], 'hosp', acc)

            ts, dead_mu, dead_sig = self.__comp_state_over_time(
                sims[i], 'dead', acc)

            # Convert x-axis into posix timestamps and use pandas to plot as dates
            ts = days_to_datetime(ts, start_date=start_date)

            # lines
            # ax.errorbar(ts, hosp_mu, yerr=2*hosp_sig, label=titles[i], errorevery=errorevery,
            #             c=self.color_different_scenarios[i], linestyle='-', elinewidth=0.8, capsize=3.0)

            # ax.errorbar(ts, dead_mu, yerr=2*dead_sig, errorevery=errorevery,
            #             c=self.color_different_scenarios[i], linestyle='dotted', elinewidth=0.8, capsize=3.0)

            ax.plot(ts, hosp_mu, linestyle='-',
                    label=titles[i], c=self.color_different_scenarios[i])
            ax.fill_between(ts, hosp_mu - 2 * hosp_sig, hosp_mu + 2 * hosp_sig,
                            color=self.color_different_scenarios[i], alpha=self.filling_alpha, linewidth=0.0)
            
            ax.plot(ts, dead_mu, linestyle='dotted', c=self.color_different_scenarios[i])
            ax.fill_between(ts, dead_mu - 2 * dead_sig, dead_mu + 2 * dead_sig,
                            color=self.color_different_scenarios[i], alpha=self.filling_alpha, linewidth=0.0)



        # axis
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + ipre_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')

        if not isinstance(lockdown_at, dict):
            if lockdown_at is not None:
                lockdown_widget(lockdown_at, start_date,
                                lockdown_label_y, ymax,
                                lockdown_label, ax, xshift=0.5)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #set ticks every week
        # ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        if show_legend:
            # legend
            if legend_is_left:
                leg = ax.legend(loc='upper left', borderaxespad=0.5)
            else:
                leg = ax.legend(loc='upper right', borderaxespad=0.5)

            if legendYoffset != 0.0:
                # Get the bounding box of the original legend
                bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

                # Change to location of the legend.
                bb.y0 += legendYoffset
                bb.y1 += legendYoffset
                leg.set_bbox_to_anchor(bb, transform=ax.transAxes)

            if legendXoffset != 0.0:
                # Get the bounding box of the original legend
                bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

                # Change to location of the legend.
                bb.x0 += legendXoffset
                bb.x1 += legendXoffset
                leg.set_bbox_to_anchor(bb, transform=ax.transAxes)

        subplot_adjust = subplot_adjust or {
            'bottom': 0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_2d_infections_at_time(self, sim, at_time, density_bandwidth=1.0, restart=0,
        title='Example', filename='2d_inf_0', figsize=(10, 10), acc=1000, ymax=None):

        '''
        Plots 2d visualization using mobility object. The bandwidth set by `density_bandwidth`
        determines the bandwidth of the RBF kernel in KDE used to generate the plot.
        Smaller means more affected by local changes. Set the colors and markers in the __init__ function
        '''
        if acc > sim.max_time:
            acc = int(sim.max_time)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # infections
        r = restart
        is_expo = self.__is_state_at(sim, r, 'expo', at_time)
        is_iasy = self.__is_state_at(sim, r, 'iasy', at_time)
        is_ipre = self.__is_state_at(sim, r, 'ipre', at_time)
        is_isym = self.__is_state_at(sim, r, 'isym', at_time)
        is_infected = is_iasy | is_ipre | is_isym
        no_state = (1 - is_infected) & (1 - is_expo)

        idx_expo = np.where(is_expo)[0]
        idx_infected = np.where(is_infected)[0]
        idx_none = np.where(no_state)[0]

        # self.color_isym = 'red'
        # self.color_expo= 'yellow'


        ### sites
        site_loc = sim.site_loc
        ax.scatter(site_loc[:, 0], site_loc[:, 1], alpha=self.filling_alpha, label='public sites',
                   marker=self.marker_site, color=self.color_site, facecolors=self.color_site, s=self.size_site)


        ### home locations and their states
        home_loc = sim.home_loc
        # no state
        ax.scatter(home_loc[idx_none, 0], home_loc[idx_none, 1],
                   marker=self.marker_home, color=self.color_home,
                   facecolors='none', s=self.size_home)

        try:
            # expo
            ax.scatter(home_loc[idx_expo, 0], home_loc[idx_expo, 1],
                    marker=self.marker_home, color=self.color_home,
                    facecolors=self.color_expo, s=self.size_home, label='exposed households')
            sns.kdeplot(home_loc[idx_expo, 0], home_loc[idx_expo, 1], shade=True, alpha=self.density_alpha,
                        shade_lowest=False, cbar=False, ax=ax, color=self.color_expo, bw=density_bandwidth, zorder=0)

            # infected
            ax.scatter(home_loc[idx_infected, 0], home_loc[idx_infected, 1],
                    marker=self.marker_home, color=self.color_home,
                    facecolors=self.color_isym, s=self.size_home, label='infected households')
            sns.kdeplot(home_loc[idx_infected, 0], home_loc[idx_infected, 1], shade=True, alpha=self.density_alpha,
                        shade_lowest=False, cbar=False, ax=ax, color=self.color_isym, bw=density_bandwidth, zorder=0)

        except:
            print('KDE failed, likely no exposed and infected at this time. Try different timing.')
            plt.close()
            return

        # axis
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        plt.axis('off')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.85)

        ax.set_title(title, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def compare_hospitalizations_over_time(self, sims, titles, figtitle='Hospitalizations', filename='compare_hosp_0',
        capacity_line_at=20, figsize=(10, 10), errorevery=20, acc=1000, ymax=None):
        ''''
        Plots total hospitalizations for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev.
        The value of `capacity_line_at` defines the y-intercept of the hospitalization capacity line
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for i in range(len(sims)):
            if acc > sims[i].max_time:
                acc = int(sims[i].max_time)

            ts, line_hosp, error_sig = self.__comp_state_over_time(
                sims[i], 'hosp', acc)
            line_xaxis = np.zeros(ts.shape)

            # lines
            ax.errorbar(ts, line_hosp, yerr=error_sig, errorevery=errorevery,
                        c='black', linestyle='-', elinewidth=0.8)

            # filling
            ax.fill_between(ts, line_xaxis, line_hosp, alpha=self.filling_alpha, zorder=0,
                            label=r'Hospitalized under: ' + titles[i], edgecolor=self.color_different_scenarios[i],
                            facecolor=self.color_different_scenarios[i], linewidth=0)

        # capacity line
        ax.plot(ts, capacity_line_at * np.ones(ts.shape[0]), label=r'Max. hospitalization capacity',
                    c='red', linestyle='--', linewidth=4.0)

        # axis
        ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(line_hosp + error_sig)
        ax.set_ylim((0, ymax))

        ax.set_xlabel(r'$t$ [days]')
        ax.set_ylabel(r'[people]')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # legend
        fig.legend(loc='center right', borderaxespad=0.1)
        # Adjust the scaling factor to fit your legend text completely outside the plot
        plt.subplots_adjust(right=0.70)
        ax.set_title(figtitle, pad=20)
        plt.draw()
        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return

    def plot_positives_vs_target(self, sims, titles, targets, title='Example',
        filename='inference_0', figsize=(6, 5), errorevery=1, acc=17, ymax=None,
        start_date='1970-01-01', lockdown_label='Lockdown', lockdown_at=None,
        lockdown_label_y=None, subplot_adjust=None):
        ''''
        Plots daily tested averaged over random restarts, using error bars for std-dev
        together with targets from inference
        '''

        if acc > sims[0].max_time:
            acc = int(sims[0].max_time)

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(len(sims)):
            if acc > sims[i].max_time:
                acc = int(sims[i].max_time)

            ts, posi_mu, posi_sig = self.__comp_state_over_time(
                sims[i], 'posi', acc)

            # Convert x-axis into posix timestamps and use pandas to plot as dates
            ts = days_to_datetime(ts, start_date=start_date)

            # lines
            ax.plot(ts, posi_mu, linestyle='-',
                    label=titles[i], c=self.color_different_scenarios[i])
            ax.fill_between(ts, posi_mu - 2 * posi_sig, posi_mu + 2 * posi_sig,
                            color=self.color_different_scenarios[i], alpha=self.filling_alpha, linewidth=0.0)

        # target   
        target_widget(targets, start_date, ax)


        # axis
        #ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(posi_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('Positive cases')

        if lockdown_at is not None:
            lockdown_widget(lockdown_at, start_date,
                            lockdown_label_y, ymax,
                            lockdown_label, ax)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')

        #set ticks every week
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        ax.legend(loc='upper left', borderaxespad=0.5)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.draw()

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI)#, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_age_group_positives_vs_target(self, sim, targets, ytitle=None,
                                 filename='inference_0', figsize=(6, 5), errorevery=1, acc=17, ymax=None,
                                 start_date='1970-01-01', lockdown_label='Lockdown', lockdown_at=None,
                                 lockdown_label_y=None, subplot_adjust=None):
        
        ''''
        Plots daily tested averaged over random restarts, using error bars for std-dev
        together with targets from inference
        '''

        if acc > sim.max_time:
            acc = int(sim.max_time)

        n_age_groups = targets.shape[1]
        fig, axs = plt.subplots(1, n_age_groups, figsize=figsize)

        for age in range(n_age_groups):

            # automatically shifted by `test_lag` in the function
            ts, posi_mu, posi_sig = self.__comp_state_over_time_per_age(
                sim, 'posi', acc, age)

            T = posi_mu.shape[0]

            xx = days_to_datetime(ts, start_date=start_date)
            axs[age].plot(xx, posi_mu, c='k', linestyle='-',
                    label='COVID-19 simulated case data')
            axs[age].fill_between(xx, posi_mu - posi_sig, posi_mu + posi_sig,
                            color='grey', alpha=0.1, linewidth=0.0)

            # target
            target_widget(targets[:, age], start_date, axs[age])

            # axis
            #ax.set_xlim((0, np.max(ts)))
            if ymax is None:
                ymax = 1.5 * np.max(posi_mu)
            axs[age].set_ylim((0, ymax))

            # ax.set_xlabel('Days')
            if age == 0:
                if ytitle is not None:
                    axs[age].set_ylabel(ytitle)

            axs[age].set_title(f'{age}')

            if lockdown_at is not None:
                lockdown_widget(lockdown_at, start_date,
                                lockdown_label_y, ymax,
                                lockdown_label, axs[age])

            # Hide the right and top spines
            axs[age].spines['right'].set_visible(False)
            axs[age].spines['top'].set_visible(False)
            axs[age].spines['left'].set_visible(False)
            axs[age].spines['bottom'].set_visible(False)
            axs[age].get_xaxis().set_ticks([])

            # Only show ticks on the left and bottom spines
            # axs[age].yaxis.set_ticks_position('left')

            #set ticks every week
            # axs[age].xaxis.set_major_locator(mdates.WeekdayLocator())
            #set major ticks format
            # axs[age].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            # fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        # legend
        # axs[age].legend(loc='upper left', borderaxespad=0.5)

        subplot_adjust = subplot_adjust or {
            'bottom': 0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.draw()

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI)  # , bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_daily_rts(self, sims, filename, start_date, titles=None, sigma=None,
                       r_t_range=R_T_RANGE, window=3, figsize=(6, 5),
                       subplot_adjust=None, lockdown_label='Lockdown',
                       lockdown_at=None, lockdown_label_y=None, ymax=None,
                       colors=['grey'], fill_between=True, draw_dots=True,
                       errorevery=1, show_legend=False, xtick_interval=1, ci=0.9):

        # If a single summary is provided
        if not isinstance(sims, list):
            sims = [sims]
            sigma = [sigma]

        results = list()
        for i, sim in enumerate(sims):
            res = compute_daily_rts(sim, start_date, sigma[i], r_t_range, window, ci)
            results.append(res)

        # Colors
        ABOVE = [1,0,0]
        MIDDLE = [1,1,1]
        BELOW = [0,0,0]
        cmap = ListedColormap(np.r_[
            np.linspace(BELOW,MIDDLE,25),
            np.linspace(MIDDLE,ABOVE,25)
        ])
        color_mapped = lambda y: np.clip(y, .5, 1.5)-.5

        ymax_computed = 0.0  # Keep track of max y to set limit

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for i, result in enumerate(results):
            index = result['ML'].index
            values = result['ML'].values

            # Plot dots and line
            ax.plot(index, values, c=colors[i], zorder=1, alpha=1.0)

            if draw_dots:
                ax.scatter(index, values, s=40, lw=0.0,
                           c=cmap(color_mapped(values)),
                           edgecolors='k', zorder=2)

            # Aesthetically, extrapolate credible interval by 1 day either side
            lowfn = interp1d(date2num(index), result[f'Low_{ci*100:.0f}'].values,
                            bounds_error=False, fill_value='extrapolate')
            highfn = interp1d(date2num(index), result[f'High_{ci*100:.0f}'].values,
                            bounds_error=False, fill_value='extrapolate')
            extended = pd.date_range(start=index[0], end=index[-1])
            error_low = lowfn(date2num(extended))
            error_high =  highfn(date2num(extended))

            if fill_between:
                ax.fill_between(extended, error_low, error_high,
                                color=colors[i], alpha=0.1, linewidth=0.0)
            else:
                # Ignore first value which is just prior, not informed by data
                ax.errorbar(x=index[1:], y=values[1:], label=titles[i],
                            yerr=np.vstack((result[f'Low_{ci*100:.0f}'], result[f'High_{ci*100:.0f}']))[:,1:],
                            color=colors[i], linewidth=1.0,
                            elinewidth=0.8, capsize=3.0,
                            errorevery=errorevery)

            ymax_computed = max(ymax_computed, np.max(error_high))

            # Plot horizontal line at R_t = 1
            ax.axhline(1.0, c='k', lw=1, alpha=.25);

        # limits
        ymax = ymax or 1.2 * ymax_computed
        ax.set_ylim((0, ymax_computed))

        if show_legend:
            ax.legend(loc='upper left', borderaxespad=0.5)

        # extra
        if lockdown_at is not None:
            lockdown_widget(lockdown_at, start_date,
                            lockdown_label_y, ymax,
                            lockdown_label, ax, zorder=-200)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Set label
        ax.set_ylabel(r'$R_t$')

        #set ticks every week
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=xtick_interval))
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI)#, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
