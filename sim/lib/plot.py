import os
import itertools
import collections
import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.colors import ListedColormap
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import FormatStrFormatter

from scipy.interpolate import griddata
import scipy.stats
import matplotlib.colors as colors

from lib.calibrationFunctions import downsample_cases, pdict_to_parr, load_state
from lib.data import collect_data_from_df

import botorch.utils.transforms as transforms

from lib.calibrationSettings import (
    calibration_model_param_bounds_single,
    calibration_start_dates,
    calibration_lockdown_dates,
    calibration_mob_paths,
)
from lib.calibrationFunctions import (
    pdict_to_parr,
    load_state,
    downsample_cases,
    CORNER_SETTINGS_SPACE,
)

from lib.data import collect_data_from_df

import botorch.utils.transforms as transforms

from lib.calibrationSettings import (
    calibration_model_param_bounds_single,
    calibration_start_dates,
    calibration_lockdown_dates,
    calibration_mob_paths,
)

from lib.rt import compute_daily_rts, R_T_RANGE
from lib.rt_nbinom import overdispersion_test
from lib.summary import *

TO_HOURS = 24.0
DPI = 200
NO_PLOT = False
TEST_LAG = 48.0 # hours

LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_TRIPLE = (COL_WIDTH / 3, COL_WIDTH / 3 * 4/6)
FIG_SIZE_TRIPLE_TALL = (COL_WIDTH / 3, COL_WIDTH / 3 * 5/6)

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
FIG_SIZE_DOUBLE_TALL = (COL_WIDTH / 2, COL_WIDTH / 2 * 5/6)

CUSTOM_FIG_SIZE_FULL_PAGE_TRIPLE = (LINE_WIDTH / 3, COL_WIDTH / 2 * 5/6)

FIG_SIZE_FULL_PAGE_TRIPLE = (LINE_WIDTH / 3, LINE_WIDTH / 3 * 4/6)
FIG_SIZE_FULL_PAGE_TRIPLE_TALL = (LINE_WIDTH / 3, LINE_WIDTH / 3 * 5/6)
FIG_SIZE_FULL_PAGE_DOUBLE_ARXIV = (LINE_WIDTH / 2, LINE_WIDTH / 3 * 4/6) # 2
FIG_SIZE_FULL_PAGE_DOUBLE_ARXIV_TALL = (LINE_WIDTH / 2, LINE_WIDTH / 3 * 4.5/6) # 2 tall
FIG_SIZE_FULL_PAGE_TRIPLE_ARXIV = (LINE_WIDTH / 3.3, LINE_WIDTH / 3 * 3.5/6) # 4x3 full page
FIG_SIZE_FULL_PAGE_TRIPLE_ARXIV_SMALL = (LINE_WIDTH / 3.7, LINE_WIDTH / 3 * 2.5/6) # 6x4 full page
CUSTOM_FIG_SIZE_FULL_PAGE_QUAD = (LINE_WIDTH / 4, COL_WIDTH / 2 * 5/6)

SIGCONF_RCPARAMS_DOUBLE = {
    # Fig params
    "figure.autolayout": True,          # Makes sure nothing the feature is neat & tight.
    "figure.figsize": FIG_SIZE_DOUBLE,  # Column width: 3.333 in, space between cols: 0.333 in.
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    # Axes params
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    'xtick.major.pad': 1.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,
    "axes.labelpad": 0.5,
    # Plot params
    "lines.linewidth": 0.8,              # Width of lines
    "lines.markeredgewidth": 0.3,
    # Legend params
    "legend.fontsize": 8.5,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # Font params
    "text.usetex": True,                 # use LaTeX to write all text
    "font.family": "serif",              # use serif rather than sans-serif
    "font.serif": "Linux Libertine O",   # use "Linux Libertine" as the standard font
    "font.size": 9,
    "axes.titlesize": 8,          # LaTeX default is 10pt font.
    "axes.labelsize": 8,          # LaTeX default is 10pt font.
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    # PDF settings
    "pgf.texsystem": "xelatex",         # Use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\usepackage{libertine}',
        r'\setmainfont{Linux Libertine O}',
        r'\setmathfont{Linux Libertine O}',
    ]
}

SIGCONF_RCPARAMS_TRIPLE = {
    # Fig params
    "figure.autolayout": True,          # Makes sure nothing the feature is neat & tight.
    "figure.figsize": FIG_SIZE_TRIPLE,  # Column width: 3.333 in, space between cols: 0.333 in.
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    # Axes params
    "axes.linewidth": 0.4,              # Matplotlib's current default is 0.8.
    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.4,
    "xtick.minor.width": 0.4,
    'xtick.major.pad': 1.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,
    "ytick.major.width": 0.4,
    "ytick.minor.width": 0.4,
    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,
    "axes.labelpad": 0.5,
    # Plot params
    "lines.linewidth": 0.8,              # Width of lines
    "lines.markeredgewidth": 0.3,
    # Legend
    "legend.fontsize": 5.5,              # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.0,
    "legend.labelspacing": 0.05,
    "patch.linewidth": 0.3,
    # Font params
    "text.usetex": True,                 # use LaTeX to write all text
    "font.family": "serif",              # use serif rather than sans-serif
    "font.serif": "Linux Libertine O",   # use "Linux Libertine" as the standard font
    "font.size": 6,
    "axes.titlesize": 5,                 # LaTeX default is 10pt font.
    "axes.labelsize": 5,                 # LaTeX default is 10pt font.
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    # PDF settings
    "pgf.texsystem": "xelatex",          # Use Xelatex which is TTF font aware
    "pgf.rcfonts": False,                # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\usepackage{libertine}',
        r'\setmainfont{Linux Libertine O}',
        r'\setmathfont{Linux Libertine O}',
    ]
}

NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)
FIG_SIZE_NEURIPS_TRIPLE = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 4/6)
FIG_SIZE_NEURIPS_DOUBLE_TALL = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 5/6)
FIG_SIZE_NEURIPS_TRIPLE_TALL = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 5/6)

NEURIPS_RCPARAMS = {
    "figure.autolayout": False,         # Makes sure nothing the feature is neat & tight.
    "figure.figsize": FIG_SIZE_NEURIPS_DOUBLE,
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    # Axes params
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,

    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    'xtick.major.pad': 1.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,

    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,

    "axes.labelpad": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    # Plot params
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    'errorbar.capsize': 3.0,
    # Font
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 8.5,
    "axes.titlesize": 8.5,                # LaTeX default is 10pt font.
    "axes.labelsize": 8.5,                # LaTeX default is 10pt font.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 7,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # PDF
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\setmainfont{Times New Roman}',
    ],
}



def trans_data_to_axis(ax):
    """Compute the transform from data to axis coordinate system in axis `ax`"""
    axis_to_data = ax.transAxes + ax.transData.inverted()
    data_to_axis = axis_to_data.inverted()
    return data_to_axis

def days_to_datetime(arr, start_date):
    # timestamps
    ts = arr * 24 * 60 * 60 + pd.Timestamp(start_date).timestamp()
    return pd.to_datetime(ts, unit='s')


def lockdown_widget(ax, lockdown_at, start_date, lockdown_label_y, lockdown_label='Lockdown',
                    xshift=0.0, zorder=None, ls='--', color='black', text_off=False):
    """
    Draw the lockdown widget corresponding to a vertical line at the desired location along with a
    label. The data can be passed either in `float` or in `datetime` format.

    Parameters
    ----------
    ax
        Axis to draw on
    lockdown_at
        Location of vertical lockdown line
    start_date
        Value of the origin of the x-axis
    lockdown_label_y
        Location of the text label on the y-axis
    lockdown_label : str (optional, default: 'Lockdown')
        Text label
    xshift : float (optional, default: 0.0)
        Shift in a-axis of the text label
    zorder : int (optional, default: None)
        z-order of the widget
    ls : str (optional, default: '--')
        Linestyle of the vertical line
    color : str (optional, default: 'black')
        color of the vertical line
    text_off : bool (optional, default: False)
        Indicate if the text label should be turned off
    """
    if isinstance(start_date, float):  # If plot with float x-axis
        lckdn_x = start_date + lockdown_at
        ax.axvline(lckdn_x, linestyle=ls, color=color, label='_nolegend_',
                   zorder=zorder)
    else:
        # If plot with datetime x-axis
        lckdn_dt = days_to_datetime(lockdown_at, start_date=start_date)  # str to datetime
        lckdn_x_d = lckdn_dt.toordinal()  # datetime to float in data coordinates
        ax.axvline(lckdn_x_d, linestyle=ls, color=color, label='_nolegend_',
                   zorder=zorder)
        # Display the text label
        if not text_off:
            if xshift == 0.0:
                # Automatic shift of the text in the plot (normalized) axis coordinates
                lckdn_x_a, _ = trans_data_to_axis(ax).transform([lckdn_x_d, 0.0])  # data coordinates to axis coordinates
                ax.text(x=lckdn_x_a, y=lockdown_label_y, s=lockdown_label,
                        transform=ax.transAxes, rotation=90,
                        verticalalignment='bottom',
                        horizontalalignment='right')
            else:
                # NOTE: for backward-compatibility, manual shift of the text, should be removed
                ax.text(x=lckdn_dt + pd.Timedelta(xshift, unit='d'),
                        y=lockdown_label_y, s=lockdown_label, rotation=90)

def target_widget(show_target,start_date, ax, zorder=None, ms=4.0, label='COVID-19 case data'):
    txx = np.linspace(0, show_target.shape[0] - 1, num=show_target.shape[0])
    txx = days_to_datetime(txx, start_date=start_date)
    ax.plot(txx, show_target, ls='', marker='x', ms=ms,
            color='black', label=label, zorder=zorder)


class CustomSitesProportionFixedLocator(plt.Locator):
    """
    Custom locator to avoid tick font bug of matplotlib
    """

    def __init__(self):
        pass

    def __call__(self):
        return np.log(np.array([2, 5, 10, 25, 100]))


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

        self.color_posi = '#4daf4a'
        self.color_nega = '#e41a1c'

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
            #'#984ea3',
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

        self.color_model_fit_light =  '#bdbdbd'
        self.color_model_fit_dark =  '#636363'

        # 2D visualization
        self.density_alpha = 0.7

        self.marker_home = "^"
        self.marker_site = "o"

        self.color_home = '#000839'
        self.color_site = '#000000'

        self.size_home = 80
        self.size_site = 300

    def _set_matplotlib_params(self, format='double'):
        # matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
        if format == 'double':
            plt.rcParams.update(SIGCONF_RCPARAMS_DOUBLE)
        elif format == 'triple':
            plt.rcParams.update(SIGCONF_RCPARAMS_TRIPLE)
        if format == 'neurips-double':
            plt.rcParams.update(NEURIPS_RCPARAMS)
        else:
            raise ValueError('Invalid figure format.')

    def _set_default_axis_settings(self, ax):
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

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
            lockdown_widget(ax, lockdown_at, start_date,
                            lockdown_label_y,
                            lockdown_label)
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

        ts, iasy_mu, iasy_sig = comp_state_over_time(sim, 'iasy', acc)
        _,  ipre_mu, ipre_sig = comp_state_over_time(sim, 'ipre', acc)
        _,  isym_mu, isym_sig = comp_state_over_time(sim, 'isym', acc)
        # _,  expo_mu, iexpo_sig = comp_state_over_time(sim, 'expo', acc)
        # _,  posi_mu, posi_sig = comp_state_over_time(sim, 'posi', acc)

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
            lockdown_widget(ax, lockdown_at, start_date,
                            lockdown_label_y,
                            lockdown_label)
        if lockdown_end is not None:
            lockdown_widget(ax=ax, lockdown_at=lockdown_end, start_date=start_date,
                            lockdown_label_y=lockdown_label_y,
                            lockdown_label='End of lockdown', ls='dotted')
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
        ts, posi_mu, posi_sig = comp_state_over_time(sim, 'posi', acc)
        _,  nega_mu, nega_sig = comp_state_over_time(sim, 'nega', acc)

        line_xaxis = np.zeros(ts.shape)
        line_posi = posi_mu
        line_nega = posi_mu + nega_mu

        error_posi = posi_sig
        error_nega = nega_sig + posi_sig

        T = posi_mu.shape[0]

        # lines
        ax.errorbar(ts, line_posi, yerr=posi_sig, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='dotted')
        ax.errorbar(ts, line_nega, yerr=nega_sig, elinewidth=0.8, errorevery=errorevery,
                c='black', linestyle='-')

        # filling
        ax.fill_between(ts, line_xaxis, line_posi, alpha=0.5, label=r'Positive tests',
                        edgecolor=self.color_posi, facecolor=self.color_posi, linewidth=0, zorder=0)
        ax.fill_between(ts, line_posi, line_nega, alpha=0.5, label=r'Negative tests',
                        edgecolor=self.color_nega, facecolor=self.color_nega, linewidth=0, zorder=0)
        # axis
        ax.set_xlim((0, np.max(ts)))
        if ymax is None:
            ymax = 1.5 * np.max(posi_mu + nega_mu)
        ax.set_ylim((0, ymax))

        ax.set_xlabel(r'$t$ [days]')
        ax.set_ylabel(r'Tests')

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

        ts, all_mu, all_sig = comp_contained_over_time(sim, 'SocialDistancingForAllMeasure', acc)
        _,  positive_mu, positive_sig = comp_contained_over_time(sim, 'SocialDistancingForPositiveMeasure', acc)
        _,  age_mu, age_sig = comp_contained_over_time(sim, 'SocialDistancingByAgeMeasure', acc)
        _,  tracing_mu, tracing_sig = comp_contained_over_time(sim, 'SocialDistancingForSmartTracing', acc)

        _, iasy_mu, iasy_sig = comp_state_over_time(sim, 'iasy', acc)
        _,  ipre_mu, ipre_sig = comp_state_over_time(sim, 'ipre', acc)
        _,  isym_mu, isym_sig = comp_state_over_time(sim, 'isym', acc)

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

    def compare_total_infections(self, sims, titles, figtitle='Title', figformat='double',
        filename='compare_inf_0', figsize=None, errorevery=20, acc=500, ymax=None, x_axis_dates=True,
        lockdown_label='Lockdown', lockdown_at=None, lockdown_label_y=None, lockdown_xshift=0.0,
        conditional_measures=None,
        show_positives=False, show_legend=True, legend_is_left=False,
        subplot_adjust=None, start_date='1970-01-01', xtick_interval=2, first_one_dashed=False,
        show_single_runs=False, which_single_runs=None):
        ''''
        Plots total infections for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev
        '''

        assert isinstance(sims[0], str), '`sims` must be list of filepaths'

        # Set double figure format
        self._set_matplotlib_params(format=figformat)
        # Draw figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for i, sim in enumerate(sims):
            is_conditional = True if i == conditional_measures else False
            try:
                data = load_condensed_summary(sim, acc)
            except FileNotFoundError:
                acc = create_condensed_summary_from_path(sim, acc=acc)
                data = load_condensed_summary(sim, acc)

            ts = data['ts']
            lockdown_at = data['lockdowns'] if is_conditional else lockdown_at
            if x_axis_dates:
                # Convert x-axis into posix timestamps and use pandas to plot as dates
                ts = days_to_datetime(ts, start_date=start_date)

            if not show_single_runs:
                iasy_mu = data['iasy_mu']
                iasy_sig = data['iasy_sig']
                ipre_mu = data['ipre_mu']
                ipre_sig = data['ipre_sig']
                isym_mu = data['isym_mu']
                isym_sig = data['isym_sig']

                line_infected = iasy_mu + ipre_mu + isym_mu
                error_infected = np.sqrt(np.square(iasy_sig) + np.square(ipre_sig) + np.square(isym_sig))

                # lines
                ax.plot(ts, line_infected, linestyle='-', label=titles[i], c=self.color_different_scenarios[i])
                ax.fill_between(ts, np.maximum(line_infected - 2 * error_infected, 0), line_infected + 2 * error_infected,
                                color=self.color_different_scenarios[i], alpha=self.filling_alpha, linewidth=0.0)
            else:
                iasy = data['iasy']
                ipre = data['ipre']
                isym = data['isym']

                lines_infected = iasy + ipre + isym

                # lines
                runs = [which_single_runs] if which_single_runs else range(min(show_single_runs, sim.random_repeats))
                for k, r in enumerate(runs):
                    ax.plot(ts, lines_infected[:, r], linestyle='-', label=titles[i] if k == 0 else None,
                            c=self.color_different_scenarios[i])

                    # For conditional measures only
                    if lockdown_at:
                        for lockdown in lockdown_at[r]:
                            start_lockdown = lockdown[0] / TO_HOURS
                            end_lockdown = lockdown[1] / TO_HOURS
                            lockdown_widget(ax, start_lockdown, 0.0,
                                            lockdown_label_y,
                                            None)
                            lockdown_widget(ax, end_lockdown, 0.0,
                                            lockdown_label_y,
                                            None, ls='-')

        # axis
        ax.set_xlim(left=np.min(ts))
        if ymax is None:
            ymax = 1.5 * np.max(iasy_mu + ipre_mu + isym_mu)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        if x_axis_dates:
            # set xticks every week
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=1, interval=1))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1, interval=xtick_interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')
        else:
            ax.set_xlabel(r'$t$ [days]')

        ax.set_ylabel('Infected')

        if not isinstance(lockdown_at, list):
            if lockdown_at is not None:
                lockdown_widget(ax, lockdown_at, start_date,
                                lockdown_label_y,
                                lockdown_label,
                                xshift=lockdown_xshift)

        # Set default axes style
        self._set_default_axis_settings(ax=ax)

        if show_legend:
            # legend
            if legend_is_left:
                leg = ax.legend(loc='upper left',
                          bbox_to_anchor=(0.001, 0.999),
                          bbox_transform=ax.transAxes,
                        #   prop={'size': 5.6}
                          )
            else:
                leg = ax.legend(loc='upper right',
                          bbox_to_anchor=(0.999, 0.999),
                          bbox_transform=ax.transAxes,
                        #   prop={'size': 5.6}
                          )

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def compare_quantity(self, sims, labels, titles=None, quantity='infected', mode='total', ymax=None, colors=None,
                         start_date='1970-01-01', xtick_interval=3, x_axis_dates=False,
                         figformat='double', filename='compare_epidemics', figsize=None,
                         lockdown_label='Lockdown', lockdown_at=None, lockdown_label_y=None, lockdown_xshift=0.0,
                         show_legend=True, legend_is_left=False, subplot_adjust=None):
        ''''
        Plots `quantity` in `mode` for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev
        '''

        #assert isinstance(sims[0], str), '`sims` must be list of filepaths'
        if isinstance(sims[0], str):
            sims = [sims]
            titles = [titles]
            multiplot = False
        else:
            multiplot = True
        assert mode in ['total', 'daily', 'cumulative', 'weekly incidence']
        assert quantity in ['infected', 'hosp', 'dead']

        labeldict = {'total': {'infected': 'Infected',
                               'hosp': 'Hospitalized',
                               'dead': 'Fatalities'},
                     'cumulative': {'infected': 'Cumulative Infections',
                                    'hosp': 'Cumulative Hospitalizations',
                                    'dead': 'Cumulative Fatalities'},
                     'daily': {'infected': 'Daily Infections',
                               'hosp': 'Daily Hospitalizations',
                               'dead': 'Daily Fatalities'},
                     'weekly incidence': {'infected': 'Weekly infection incidence'}
                     }

        # Set double figure format
        self._set_matplotlib_params(format=figformat)
        # Draw figure
        fig, axs = plt.subplots(len(sims), 1, figsize=figsize)
        if not multiplot:
            axs = [axs]
        for j, (paths, ax) in enumerate(zip(sims, axs)):
            for i, sim in enumerate(paths):
                data = load_condensed_summary_compat(sim)
                line_cases, error_cases = get_plot_data(data, quantity=quantity, mode=mode)

                if mode in ['daily', 'weekly incidence']:
                    ts = np.arange(0, len(line_cases))
                    if mode == 'daily':
                        error_cases = np.zeros(len(line_cases))
                else:
                    ts = data['ts'] if not x_axis_dates else days_to_datetime(data['ts'], start_date=start_date)

                if colors is None:
                    colors = self.color_different_scenarios[i]

                # lines
                ax.plot(ts, line_cases, linestyle='-', label=labels[i], c=colors[i])
                ax.fill_between(ts, np.maximum(line_cases - 2 * error_cases, 0), line_cases + 2 * error_cases,
                                color=colors[i], alpha=self.filling_alpha, linewidth=0.0)

            # axis
            ax.set_xlim(left=np.min(ts))
            if ymax is None:
                ymax = 1.5 * np.max(line_cases)
            ax.set_ylim((0, ymax))
            if titles:
                ax.set_title(titles[j])

            if x_axis_dates:
                # set xticks every week
                ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=1, interval=1))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1, interval=xtick_interval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')
            else:
                ax.set_xlabel(r'$t$ [days]')

            ylabel = labeldict[mode][quantity]
            ax.set_ylabel(ylabel)

            if lockdown_at is not None:
                lockdown_widget(ax, lockdown_at, start_date,
                                lockdown_label_y,
                                lockdown_label,
                                xshift=lockdown_xshift)

            # Set default axes style
            self._set_default_axis_settings(ax=ax)

            if show_legend:
                # legend
                if legend_is_left:
                    leg = ax.legend(loc='upper left',
                              bbox_to_anchor=(0.001, 0.999),
                              bbox_transform=ax.transAxes,
                            #   prop={'size': 5.6}
                              )
                else:
                    leg = ax.legend(loc='upper right',
                              bbox_to_anchor=(0.999, 0.999),
                              bbox_transform=ax.transAxes,
                            #   prop={'size': 5.6}
                              )
                # if titles:
                #     axs[0].legend(loc='lower left', bbox_to_anchor=(1.1, 0.18),
                #                 borderaxespad=0, frameon=True)

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)
        if multiplot:
            plt.tight_layout()

        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def compare_total_fatalities_and_hospitalizations(self, sims, titles, mode='show_both',
        figtitle=r'Hospitalizations and Fatalities',
        lockdown_label='Lockdown', lockdown_at=None, lockdown_label_y=None,
        figformat='neurips-double',
        xtick_interval=2, lockdown_xshift=0.0,
        filename='compare_inf_0', figsize=(10, 10), errorevery=20, acc=1000, ymax=None,
        show_legend=True, legendYoffset=0.0, legend_is_left=False, legendXoffset=0.0,
        subplot_adjust=None, start_date='1970-01-01', first_one_dashed=False):

        ''''
        Plots total fatalities and hospitalizations for each simulation, named as provided by `titles`
        to compare different measures/interventions taken. Colors taken as defined in __init__, and
        averaged over random restarts, using error bars for std-dev
        '''

        # Set double figure format
        self._set_matplotlib_params(format=figformat)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # hospitalizations
        for i, sim in enumerate(sims):
            if isinstance(sim, str):
                try:
                    data = load_condensed_summary(sim, acc=acc)
                except FileNotFoundError:
                    acc = create_condensed_summary_from_path(sim, acc=acc)
                    data = load_condensed_summary(sim, acc=acc)

                acc = data['acc']
                ts = data['ts']
                hosp_mu = data['hosp_mu']
                hosp_sig = data['hosp_sig']
                dead_mu = data['dead_mu']
                dead_sig = data['dead_sig']
                loaded_extracted_data = True
            else:
                loaded_extracted_data = False

            if not loaded_extracted_data:
                if acc > sim.max_time:
                    acc = int(sim.max_time)

                ts, hosp_mu, hosp_sig = comp_state_over_time(sim, 'hosp', acc)
                ts, dead_mu, dead_sig = comp_state_over_time(sim, 'dead', acc)

            # Convert x-axis into posix timestamps and use pandas to plot as dates
            ts = days_to_datetime(ts, start_date=start_date)

            # lines
            # ax.errorbar(ts, hosp_mu, yerr=2*hosp_sig, label=titles[i], errorevery=errorevery,
            #             c=self.color_different_scenarios[i], linestyle='-', elinewidth=0.8, capsize=3.0)

            # ax.errorbar(ts, dead_mu, yerr=2*dead_sig, errorevery=errorevery,
            #             c=self.color_different_scenarios[i], linestyle='dotted', elinewidth=0.8, capsize=3.0)
            if mode == 'show_both' or mode == 'show_hosp_only':
                ax.plot(ts, hosp_mu, linestyle='-',
                        label=titles[i], c=self.color_different_scenarios[i])
                ax.fill_between(ts, hosp_mu - 2 * hosp_sig, hosp_mu + 2 * hosp_sig,
                                color=self.color_different_scenarios[i], alpha=self.filling_alpha, linewidth=0.0)

            if mode == 'show_both' or mode == 'show_dead_only':
                linestyle = '-' if mode == 'show_dead_only' else 'dotted'
                labels = titles[i] if mode == 'show_dead_only' else None
                ax.plot(ts, dead_mu, linestyle=linestyle,
                        label=labels, c=self.color_different_scenarios[i])
                ax.fill_between(ts, dead_mu - 2 * dead_sig, dead_mu + 2 * dead_sig,
                                color=self.color_different_scenarios[i], alpha=self.filling_alpha, linewidth=0.0)

        # axis
        ax.set_xlim(left=np.min(ts))
        if ymax is None:
            ymax = 1.5 * np.max(hosp_mu + hosp_sig)
        ax.set_ylim((0, ymax))

        # ax.set_xlabel('Days')
        ax.set_ylabel('People')

        if not isinstance(lockdown_at, list):
            if lockdown_at is not None:
                lockdown_widget(ax, lockdown_at, start_date,
                                lockdown_label_y,
                                lockdown_label,
                                xshift=lockdown_xshift)

        # ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=1, interval=1))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1, interval=xtick_interval))
       

        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')

        self._set_default_axis_settings(ax=ax)

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

        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
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
        is_expo = is_state_at(sim, r, 'expo', at_time)
        is_iasy = is_state_at(sim, r, 'iasy', at_time)
        is_ipre = is_state_at(sim, r, 'ipre', at_time)
        is_isym = is_state_at(sim, r, 'isym', at_time)
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
        capacity_line_at=20, figsize=(10, 10), errorevery=20, acc=500, ymax=None):
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

            ts, line_hosp, error_sig = comp_state_over_time(
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

    def plot_positives_vs_target(self, paths, labels, country, area, ymax=None,
                                 lockdown_label='Interventions', lockdown_label_y=None,
                                 filename='Modelfit', show_legend=True,
                                 figsize=None, figformat='triple', small_figure=False, cluster_compatible=False):
        ''''
        Plots daily tested averaged over random restarts, using error bars for std-dev
        together with targets from inference
        '''
        # Set triple figure format
        if not cluster_compatible:
            self._set_matplotlib_params(format=figformat)

        # Get target cases for the specific region
        start_date = calibration_start_dates[country][area]
        start_date_lockdown = calibration_lockdown_dates[country]['start']
        end_date = calibration_lockdown_dates[country]['end']

        lockdown_at = (pd.to_datetime(start_date_lockdown) - pd.to_datetime(start_date)).days

        mob_settings_paths = calibration_mob_paths[country][area][1]
        with open(mob_settings_paths, 'rb') as fp:
            mob_settings = pickle.load(fp)

        area_cases = collect_data_from_df(country=country,
                                          area=area,
                                          datatype='new',
                                          start_date_string=start_date,
                                          end_date_string=end_date)

        sim_cases = downsample_cases(area_cases, mob_settings)  # only downscaling due LK data for cities
        targets = sim_cases.sum(axis=1)

        fig, ax = plt.subplots(figsize=figsize)

        for i, path in enumerate(paths):
            if isinstance(path, str):
                # path of `Result` in `experiment.py`
                data = load_condensed_summary(path)

                print('metadata.model_params:')
                print(f'.beta_household {data["metadata"].model_params["beta_household"]}')
                print(f'.beta_site      {data["metadata"].model_params["betas"]}\n')

                ts = data['ts']
                posi_mu = data['posi_mu']
                posi_sig = data['posi_sig']
            else:
                # (uncondensed) summary from `calibrate.py`
                ts, posi_mu, posi_sig = comp_state_over_time(path, 'posi', acc=500)

            # Convert x-axis into posix timestamps and use pandas to plot as dates
            plain_ts = ts
            ts = days_to_datetime(ts, start_date=start_date)

            # lines
            ax.fill_between(ts, posi_mu - 2 * posi_sig, posi_mu - 1 * posi_sig,
                            color=self.color_model_fit_light,
                            linewidth=0.0, alpha=0.7)
            ax.fill_between(ts, posi_mu + 1 * posi_sig, posi_mu + 2 * posi_sig,
                            color=self.color_model_fit_light,
                            linewidth=0.0, alpha=0.7)
            ax.fill_between(ts, posi_mu - 1 * posi_sig, posi_mu + 1 * posi_sig,
                            color=self.color_model_fit_dark, label=labels[i],
                            linewidth=0.0, alpha=0.7)
            ax.plot(ts, posi_mu, c='black')

        # target
        if small_figure:
            target_widget(targets, start_date, ax, label='Real cases', ms=1.0)
        else:
            target_widget(targets, start_date, ax, label='Real cases')
        if ymax is None:
            ymax = 1.5 * np.max(posi_mu)
        ax.set_ylim((0, ymax))
        ax.set_ylabel(r'Positive cases')

        # lockdown
        if lockdown_at is not None:
            if small_figure:
                xshift = 3.5 * pd.to_timedelta(pd.to_datetime(ts[-1]) - pd.to_datetime(start_date), 'd') / 54
                text_off = True
            else:
                xshift = 2.5 * pd.to_timedelta(pd.to_datetime(ts[-1]) - pd.to_datetime(start_date), 'd') / 54
                text_off = True
            lockdown_widget(ax, lockdown_at, start_date,
                            lockdown_label_y,
                            lockdown_label, xshift=0.0, text_off=text_off)
        # Default axes style
        self._set_default_axis_settings(ax=ax)
        # y-ticks
        if small_figure:
            if ymax > 700:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
            else:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(250))
        # x-ticks
        if small_figure:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=2, interval=4))
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        #set major ticks format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')
        
        # sort both labels and handles (so simulated cases is above real cases in legend)
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))

        # legend
        if show_legend:
            if small_figure:
                ax.legend(handles, labels, loc='upper left',
                          bbox_to_anchor=(0.025, 0.99),
                          bbox_transform=ax.transAxes,)
            else:
                ax.legend(handles, labels, loc='upper left', borderaxespad=0.5)
        # Save fig
        if cluster_compatible:
            plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        else:
            plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        if NO_PLOT:
            plt.close()
        return plain_ts, posi_mu

    def plot_age_group_positives_vs_target(self, sim, targets, ytitle=None,
                                 filename='inference_0', figsize=(6, 5), errorevery=1, acc=17, ymax=None,
                                 start_date='1970-01-01', lockdown_label='Lockdown', lockdown_at=None,
                                 lockdown_label_y=None, subplot_adjust=None):

        ''''
        Plots daily tested averaged over random restarts, using error bars for std-dev
        together with targets from inference
        '''

        n_age_groups = targets.shape[1]
        if n_age_groups == 6:
            age_groups = ['0-4', '5-15', '15-34', '35-59', '60-79', '80+']
        else:
            age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

        if isinstance(sim, str):
            try:
                data = load_condensed_summary(sim, acc=acc)
            except FileNotFoundError:
                acc = create_condensed_summary_from_path(sim, acc=acc, n_age_groups=n_age_groups)
                data = load_condensed_summary(sim, acc=acc)
        else:
            if acc > sim.max_time:
                acc = int(sim.max_time)

        fig, axs = plt.subplots(1, n_age_groups, figsize=figsize)

        for i, age in enumerate(range(n_age_groups)):

            if isinstance(sim, str):
                ts = data['ts']
                posi_mu = data['posi_mu_age'][i]
                posi_sig = data['posi_sig_age'][i]
            else:
                # automatically shifted by `test_lag` in the function
                ts, posi_mu, posi_sig = comp_state_over_time_per_age(sim, 'posi', acc, age)

            T = posi_mu.shape[0]

            xx = days_to_datetime(ts, start_date=start_date)
            axs[age].plot(xx, posi_mu, c=self.color_different_scenarios[0], linestyle='-',
                    label='COVID-19 simulated case data')
            axs[age].fill_between(xx, posi_mu - 2 * posi_sig, posi_mu + 2 * posi_sig,
                            color=self.color_different_scenarios[0], alpha=0.1, linewidth=0.0)

            # target
            target_widget(targets[:, age], start_date, axs[age], ms=4)

            # axis
            #ax.set_xlim((0, np.max(ts)))
            if ymax is None:
                ymax = 1.5 * np.max(posi_mu)
            axs[age].set_ylim((0, ymax))

            # ax.set_xlabel('Days')
            if age == 0:
                if ytitle is not None:
                    axs[age].set_ylabel(ytitle)

            axs[age].set_title(f'{age_groups[age]} years')

            if lockdown_at is not None:
                xshift = 2.5 * pd.to_timedelta(pd.to_datetime(ts[-1]) - pd.to_datetime(start_date), 'd') / 54
                lockdown_widget(axs[age], lockdown_at, start_date,
                                lockdown_label_y,
                                lockdown_label, xshift=xshift)

            # Hide the right and top spines
            axs[age].spines['right'].set_visible(False)
            axs[age].spines['top'].set_visible(False)
            axs[age].spines['left'].set_visible(False)
            axs[age].spines['bottom'].set_visible(False)
            axs[age].get_xaxis().set_ticks([])

            axs[age].set_xlabel(r'$t$')
            # axs[age].set_ylabel(r'Cases')

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
        plt.tight_layout()
        plt.draw()

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI)  # , bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_daily_rts(self, sims, filename, start_date='1970-01-01', x_axis_dates=True, titles=None, sigma=None,
                       r_t_range=R_T_RANGE, window=3, figsize=(6, 5),
                       subplot_adjust=None, lockdown_label='Lockdown',
                       lockdown_at=None, lockdown_label_y=None, ymax=None,
                       colors=['grey'], fill_between=True, draw_dots=True,
                       errorevery=1, show_legend=False, xtick_interval=2, ci=0.9):

        # If a single summary is provided
        if not isinstance(sims, list):
            sims = [sims]
            sigma = [sigma]

        results = list()
        for i, sim in enumerate([sims[0]]):
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
            if x_axis_dates:
                index = result['ML'].index
            else:
                index = np.arange(0, len(result['ML'].index))
            values = result['ML'].values

            # Plot dots and line
            ax.plot(index, values, c=colors[i], zorder=1, alpha=1.0)

            if draw_dots:
                ax.scatter(index, values, s=40, lw=0.0,
                           c=cmap(color_mapped(values)),
                           edgecolors='k', zorder=2)

            # Aesthetically, extrapolate credible interval by 1 day either side
            if x_axis_dates:
                lowfn = interp1d(date2num(index), result[f'Low_{ci*100:.0f}'].values,
                                bounds_error=False, fill_value='extrapolate')
                highfn = interp1d(date2num(index), result[f'High_{ci*100:.0f}'].values,
                                bounds_error=False, fill_value='extrapolate')

                extended = pd.date_range(start=index[0], end=index[-1])
                error_low = lowfn(date2num(extended))
                error_high = highfn(date2num(extended))
            else:
                lowfn = interp1d(index, result[f'Low_{ci * 100:.0f}'].values,
                                 bounds_error=False, fill_value='extrapolate')
                highfn = interp1d(index, result[f'High_{ci * 100:.0f}'].values,
                                  bounds_error=False, fill_value='extrapolate')

                extended = index
                error_low = lowfn(extended)
                error_high = highfn(extended)

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
            xshift = 2.5 * pd.to_timedelta(pd.to_datetime(index[-1]) - pd.to_datetime(start_date), 'd') / 54
            lockdown_widget(ax, lockdown_at, start_date,
                            lockdown_label_y,
                            lockdown_label, zorder=-200, xshift=xshift)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Set label
        ax.set_ylabel(r'$R_t$')
        if x_axis_dates:
            #set ticks every week
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=xtick_interval))
            #set major ticks format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')
        else:
            ax.set_xlabel(r'$t$ [days]')

        subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.png', format='png', facecolor=None,
                    dpi=DPI)#, bbox_inches='tight')

        if NO_PLOT:
            plt.close()

    def plot_daily_nbinom_rts(self, path, filename='daily_nbinom_rts',
                              figsize=None, figformat='double', ymax=None, label=None, small_figure=False,
                              cmap_range=(0.5, 1.5), subplots_adjust={'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96},
                              lockdown_label='Lockdown', lockdown_at=None, lockdown_label_y=None, lockdown_xshift=0.0,
                              x_axis_dates=False, xtick_interval=2, xlim=None, show_legend=True, legend_is_left=False):
        # Set this plot with double figures parameters
        self._set_matplotlib_params(format=figformat)

        assert isinstance(path, str), '`path` must be a string.'
        data = load_condensed_summary(path)
        metadata = data['metadata']
        df = data['nbinom_dist']
            
        # Format dates
        if x_axis_dates:
            # Cast time of end of interval to datetime
            df['date_end'] = days_to_datetime(
                df['t1'] / 24, start_date=metadata.start_date)
            # Aggregate results by date
            df_agg = df.groupby('date_end').agg({'Rt': ['mean', 'std'],
                                                'kt': ['mean', 'std']})
        else:
            df['time'] = df['t1'] / 24
            df_agg = df.groupby('time').agg({'Rt': ['mean', 'std'],
                                             'kt': ['mean', 'std']})
        # Build dot colormap: black to white to red
        ABOVE = [1, 0, 0]
        MIDDLE = [1, 1, 1]
        BELOW = [0, 0, 0]
        cmap_raw = ListedColormap(np.r_[
            np.linspace(BELOW, MIDDLE, 25),
            np.linspace(MIDDLE, ABOVE, 25)
        ])

        def cmap_clipped(y):
            vmin, vmax = cmap_range
            return cmap_raw((np.clip(y, vmin, vmax) - vmin) / (vmax - vmin))

        # Plot figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        y_m = df_agg.Rt['mean']
        y_std = df_agg.Rt['std']
        # Plot estimated mean values (fill +/- std) with colored dots
        plt.fill_between(df_agg.index, y_m - y_std, y_m + y_std,
                        color='lightgray', linewidth=0.0, alpha=0.5)
        plt.plot(df_agg.index, y_m, c='grey')
        plt.scatter(df_agg.index, y_m, s=4.0, lw=0.0, c=cmap_clipped(y_m),
                    edgecolors='k', zorder=100, label=label)
        # Horizotal line at R_t = 1.0
        plt.axhline(1.0, c='lightgray', zorder=-100)
        # extra
        if lockdown_at is not None:
            xshift = (2.5 * pd.to_timedelta(pd.to_datetime(df_agg.index[-1])
                      - pd.to_datetime(metadata.start_date), 'd') / 54)
            ax.axvline(pd.to_datetime(lockdown_at), c='black', ls='--',
                       label='_nolegend_', zorder=-200)
            ax.text(x=lockdown_at + pd.Timedelta(lockdown_xshift, unit='d'),
                    y=lockdown_label_y, s=lockdown_label,
                    rotation=90, #fontdict={'fontsize': 5.5}
                    )

        if x_axis_dates:
            # set xticks every week
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=1, interval=1))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1, interval=xtick_interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            fig.autofmt_xdate(bottom=0.2, rotation=0, ha='center')
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
            ax.set_xlabel(r'$t$ [days]')
        # set yticks to units
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # Set labels
        ax.set_ylabel(r'$R_t$')
        # Set limits
        ax.set_ylim(bottom=0.0, top=ymax)
        if xlim:
            ax.set_xlim(*xlim)
        # Set default axes style
        self._set_default_axis_settings(ax=ax)

        if show_legend:
            # legend
            if legend_is_left:
                leg = ax.legend(loc='upper left',
                                bbox_to_anchor=(0.001, 0.999),
                                bbox_transform=ax.transAxes,
                                )
            else:
                leg = ax.legend(loc='upper right',
                                bbox_to_anchor=(0.999, 0.999),
                                bbox_transform=ax.transAxes,
                                )

        plt.subplots_adjust(**subplots_adjust)
        # Save plot
        # fpath = f"plots/daily-nbinom-rts-{filename}.pdf"
        fpath = f'plots/{filename}.pdf'
        plt.savefig(fpath, format='pdf')
        print("Save:", fpath)
        if NO_PLOT:
            plt.close()

    def plot_daily_nbinom_rts_panel(self, paths, titles, filename='daily_nbinom_rts',
                              figsize=None, figformat='double', ymax=None, label=None, small_figure=False,
                              cmap_range=(0.5, 1.5),
                              subplots_adjust={'bottom': 0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96},
                              lockdown_label='Lockdown', lockdown_at=None, lockdown_label_y=None, lockdown_xshift=0.0,
                              x_axis_dates=False, xtick_interval=2, xlim=None, show_legend=True, legend_is_left=False):
        # Set this plot with double figures parameters
        self._set_matplotlib_params(format=figformat)
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        for i, (path, ax) in enumerate(zip(paths, axs.flat)):
            assert isinstance(path, str), '`path` must be a string.'
            data = load_condensed_summary(path)
            metadata = data['metadata']
            df = data['nbinom_dist']

            # Format dates
            if x_axis_dates:
                # Cast time of end of interval to datetime
                df['date_end'] = days_to_datetime(
                    df['t1'] / 24, start_date=metadata.start_date)
                # Aggregate results by date
                df_agg = df.groupby('date_end').agg({'Rt': ['mean', 'std'],
                                                     'kt': ['mean', 'std']})
            else:
                df['time'] = df['t1'] / 24
                df_agg = df.groupby('time').agg({'Rt': ['mean', 'std'],
                                                 'kt': ['mean', 'std']})
            # Build dot colormap: black to white to red
            ABOVE = [1, 0, 0]
            MIDDLE = [1, 1, 1]
            BELOW = [0, 0, 0]
            cmap_raw = ListedColormap(np.r_[
                                          np.linspace(BELOW, MIDDLE, 25),
                                          np.linspace(MIDDLE, ABOVE, 25)
                                      ])

            def cmap_clipped(y):
                vmin, vmax = cmap_range
                return cmap_raw((np.clip(y, vmin, vmax) - vmin) / (vmax - vmin))

            # Plot figure
            y_m = df_agg.Rt['mean']
            y_std = df_agg.Rt['std']
            # Plot estimated mean values (fill +/- std) with colored dots
            ax.fill_between(df_agg.index, y_m - y_std, y_m + y_std,
                             color='lightgray', linewidth=0.0, alpha=0.5)
            ax.plot(df_agg.index, y_m, c='grey')
            ax.scatter(df_agg.index, y_m, s=4.0, lw=0.0, c=cmap_clipped(y_m),
                        edgecolors='k', zorder=100, label=label)
            # Horizotal line at R_t = 1.0
            ax.axhline(1.0, c='lightgray', zorder=-100)
            # extra
            ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
            ax.set_xlabel(r'$t$ [days]')
            # set yticks to units
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            # Set labels
            ax.set_ylabel(r'$R_t$')
            ax.set_title(titles[i])
            # Set limits
            ax.set_ylim(bottom=0.0, top=ymax)
            if xlim:
                ax.set_xlim(*xlim)
            # Set default axes style
            self._set_default_axis_settings(ax=ax)

        # plt.subplots_adjust(**subplots_adjust)
        # Save plot
        # fpath = f"plots/daily-nbinom-rts-{filename}.pdf"
        plt.tight_layout()
        fpath = f'plots/{filename}.pdf'
        plt.savefig(fpath, format='pdf')
        print("Save:", fpath)
        if NO_PLOT:
            plt.close()

    def plot_nbinom_distributions(self, *, path, figsize=FIG_SIZE_TRIPLE_TALL, figformat='triple',
                                  label_range=[], ymax=None, filename='nbinom_dist', t0=28 * 24.0):
        """
        Plot the distribution of number of secondary cases along with their Negative-Binomial fits
        for the experiment summary in `result` for several ranges of times.
        A pre-computed dataframe `df` can also be provided
        """

        # Compute statistics
        assert isinstance(path, str), '`path` must be a string.'
        data = load_condensed_summary(path)
        metadata = data['metadata']

        df = data['nbinom_dist']
        x_range = np.arange(0, 20)

        # dispersion statistics
        df_stats = overdispersion_test(
            df=df, count_data_str='num_sec_cases',
            chi2_max_count=10) # overall sec cases (important for chi2)

        # Aggregate results by time
        df_agg = df.groupby('t0').agg({'nbinom_pmf': list,
                                    'Rt': ['mean', 'std'],
                                    'kt': ['mean', 'std']})
        # Set triple figure params
        self._set_matplotlib_params(format=figformat)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Extract data for the plot
        row_df = df.loc[df.t0 == t0]
        row_df_agg = df_agg.loc[t0]
        row_stats = df_stats.loc[df_stats.t0 == t0]
        y_nbinom = np.nanmean(np.vstack(row_df_agg['nbinom_pmf']), axis=0)

        # Plot histogram
        plt.hist(np.hstack(row_df['num_sec_cases']),
                bins=x_range, density=True,
                color='darkgray',
                align='left', width=0.8,
                label='Empirical')
                
        # Plot NB pmf
        plt.plot(x_range, y_nbinom,
                color='k',
                label=r'$\mathrm{NBin}(r, k)$')

        # Write estimates in text
        text_x = 0.999
        text_y = 0.28
        plt.text(text_x, text_y + 0.28, transform=ax.transAxes, horizontalalignment='right',
                s=r'$r ~=~' + f"{row_df_agg['Rt']['mean']:.2f} \pm ({row_df_agg['Rt']['std']:.2f})$")
        plt.text(text_x, text_y + 0.15, transform=ax.transAxes, horizontalalignment='right',
                s=r'$k ~=~' + f"{row_df_agg['kt']['mean']:.2f} \pm ({row_df_agg['kt']['std']:.2f})$")

        print('Poisson/dispersion statistics:')
        print(row_stats)
        print()
        print(f'Chi-squared test: p = {row_stats["chi2_pval"].values[0]:.8f}')
        print(f'Variance test:    p = {row_stats["vt_pval"].values[0]:.8f}')
        print()

        # Set layout and labels
        plt.ylim(top=ymax)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        # plt.xlabel('Number of Secondary Infections')
        plt.ylabel('Probability')
        plt.legend(loc='upper right')

        # Set default axis style
        self._set_default_axis_settings(ax=ax)
        plt.subplots_adjust(left=0.22, bottom=0.22, right=0.99, top=0.95)

        # Save figure
        fpath = f"plots/nbin-secondary-exposures-{filename}.pdf"
        print('Save:', fpath)
        plt.savefig(fpath)
        os.system(f'pdfcrop "${fpath}" tmp.pdf && mv tmp.pdf "${fpath}"')
        if NO_PLOT:
            plt.close()

    def plot_visit_nbinom_distributions(self, *, path, figsize=FIG_SIZE_TRIPLE_TALL, figformat='triple',
                                  label_range=[], ymax=None, filename='visit_nbinom_dist', t0=28 * 24.0):
        """
        Plot the distribution of number of secondary cases along with their Negative-Binomial fits
        for the experiment summary in `result` for several ranges of times.
        A pre-computed dataframe `df` can also be provided
        """

        # Compute statistics
        assert isinstance(path, str), '`path` must be a string.'
        data = load_condensed_summary(path)
        metadata = data['metadata']

        df = data['visit_nbinom_dist']
        x_range = np.arange(0, 20)

        # dispersion statistics
        df_stats = overdispersion_test(
            df=df, count_data_str='visit_expo_counts',
            chi2_max_count=5) # max sec cases per visit

        # Aggregate results by time
        df_agg = df.groupby('t0').agg({'nbinom_pmf': list,
                                    'Rt': ['mean', 'std'],
                                    'kt': ['mean', 'std']})
        # Set triple figure params
        self._set_matplotlib_params(format=figformat)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Extract data for the plot
        row_df = df.loc[df.t0 == t0]
        row_df_agg = df_agg.loc[t0]
        row_stats = df_stats.loc[df_stats.t0 == t0]
        y_nbinom = np.nanmean(np.vstack(row_df_agg['nbinom_pmf']), axis=0)

        # Plot histogram
        plt.hist(np.hstack(row_df['visit_expo_counts']),
                bins=x_range, density=True,
                color='darkgray',
                align='left', width=0.8,
                label='Empirical')
                
        # Plot NB pmf
        plt.plot(x_range, y_nbinom,
                color='k',
                label=r'$\mathrm{NBin}(r, k)$')

        # Write estimates in text
        text_x = 0.999
        text_y = 0.28
        plt.text(text_x, text_y + 0.28, transform=ax.transAxes, horizontalalignment='right',
                s=r'$r ~=~' + f"{row_df_agg['Rt']['mean']:.2f} \pm ({row_df_agg['Rt']['std']:.2f})$")
        plt.text(text_x, text_y + 0.15, transform=ax.transAxes, horizontalalignment='right',
                s=r'$k ~=~' + f"{row_df_agg['kt']['mean']:.2f} \pm ({row_df_agg['kt']['std']:.2f})$")

        print('Poisson/dispersion statistics:')
        print(row_stats)
        print()
        print(f'Chi-squared test: p = {row_stats["chi2_pval"].values[0]:.8f}')
        print(f'Variance test:    p = {row_stats["vt_pval"].values[0]:.8f}')
        print()

        # Set layout and labels
        plt.ylim(top=ymax)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        # plt.xlabel('Number of Secondary Infections per Visit')
        plt.ylabel('Probability')
        plt.legend(loc='upper right')

        # Set default axis style
        self._set_default_axis_settings(ax=ax)
        plt.subplots_adjust(left=0.22, bottom=0.22, right=0.99, top=0.95)

        # Save figure
        fpath = f"plots/nbin-visit-exposures-{filename}.pdf"
        print('Save:', fpath)
        plt.savefig(fpath)
        os.system(f'pdfcrop "${fpath}" tmp.pdf && mv tmp.pdf "${fpath}"')
        if NO_PLOT:
            plt.close()

    def plot_rt_over_population_infected(self, path_list, baseline_path=None, titles=None,
                               area_population=None, p_population=None, show_reduction=True, log_xscale=True, ylim=(0, 100),
                               figformat='double', filename='reff', figsize=None,
                               show_legend=True, legend_is_left=False, subplot_adjust=None):


        # Set double figure format
        self._set_matplotlib_params(format=figformat)
        # Draw figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if p_population is None:
            p_population = [5, 10, 20, 30, 40, 50, 60, 70]

        baseline_data = load_condensed_summary(baseline_path)
        baseline_rt = []
        baseline_rt_std = []
        for p in p_population:
            rt, rt_std = get_rt(baseline_data, p_infected=p/100, area_population=area_population)
            baseline_rt.append(rt)
            baseline_rt_std.append(rt_std)


        # colors = self.color_different_scenarios
        colors = ['#377eb8', '#bd0026', '#f03b20', '#fd8d3c', '#fecc5c', '#ffffb2']
        #colors = [ '#bd0026', '#f03b20', '#fd8d3c', '#fecc5c', '#377eb8',]
        zorders = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        for i, path in enumerate(path_list):
            data = load_condensed_summary(path)
            data_rt = []
            data_rt_std = []
            for p in p_population:
                rt, rt_std = get_rt(data, p_infected=p/100, area_population=area_population)
                data_rt.append(rt)
                data_rt_std.append(rt_std)

            if show_reduction:
                data_rt = (1 - np.asarray(data_rt) / baseline_rt) * 100
                data_rt_std = np.asarray(data_rt_std) / baseline_rt * 100
                ylabel = r'\% reduction of $R_{\textrm{eff}}$'
            else:
                ylabel = r'$R_{\textrm{eff}}$'

            bars = ax.errorbar(p_population, data_rt, yerr=data_rt_std, label=titles[i],
                               c=colors[i], linestyle='-', elinewidth=0.8, capsize=3.0, zorder=zorders[i])

        if log_xscale:
            ax.set_xscale('log')

        # ax.set_xlim(left=np.min(p_population), right=104)
        if ylim:
            ax.set_ylim(ymax=ylim[1], ymin=ylim[0])
        ax.set_xticks(p_population)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax.set_ylabel(ylabel)
        ax.set_xlabel('\% of population not susceptible')

        if show_legend:
            # legend
            if legend_is_left:
                leg = ax.legend(loc='upper left',
                                bbox_to_anchor=(0.001, 0.999),
                                bbox_transform=ax.transAxes,
                                )
            else:
                leg = ax.legend(loc='upper right',
                                bbox_to_anchor=(0.999, 0.999),
                                bbox_transform=ax.transAxes,
                                )

        subplot_adjust = subplot_adjust or {'bottom': 0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def plot_roc_curve(self, summaries=None, paths=None, action='isolate', figformat='double',
                       p_adoption=None, p_recall=None, p_manual_reachability=None, p_beacon=None, sitetype=None,
                       filename='roc_example', figsize=None, use_medical_labels=False, verbose=True):
        ''''
        ROC curve
        '''
        assert (p_adoption and p_recall and p_manual_reachability and p_beacon) is None or \
               (p_adoption and p_recall and p_manual_reachability and p_beacon) is not None
        assert (summaries or paths) is not None and (summaries and paths) is None, "Specify either summaries or paths"
        self._set_matplotlib_params(format=figformat)
        if isinstance(p_adoption, list):
            fig, axs = plt.subplots(1, len(p_adoption), figsize=figsize)
            ps_adoption = p_adoption
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axs = [ax]
            ps_adoption = [p_adoption] if p_adoption else [None]

        for plt_index, p_adoption in enumerate(ps_adoption):
            for i, path in enumerate(paths):
                summary = load_condensed_summary(path)
                if paths:   # If condensed summary
                    tracing_stats = summary['tracing_stats']
                else:
                    print('exposed:', np.sum(summary.state_started_at['expo'] < np.inf, axis=1).mean())
                    tracing_stats = summary.tracing_stats
                thresholds = list(tracing_stats.keys())

                if sitetype is None:
                    tracing_stats_new = dict()
                    for thres in tracing_stats.keys():
                        tracing_stats_new[thres] = tracing_stats[thres]['stats']
                    tracing_stats = tracing_stats_new
                else:
                    tracing_stats_new = dict()
                    for thres in tracing_stats.keys():
                        tracing_stats_new[thres] = tracing_stats[thres][sitetype]
                    tracing_stats = tracing_stats_new

                policies = dict()
                p_tracings = []
                colorind = 0

                for j, (name, policy) in enumerate([('SPECT', 'no_sites'), ('PanCast', 'sites')]):
                    if name == 'SPECT' or p_beacon is None:
                        ps_beacon = [None]
                    else:
                        p_beacon = np.sort(p_beacon)[::-1]
                        ps_beacon = p_beacon
                    for p_beac in ps_beacon:

                        fpr_mean, fpr_std = [], []
                        tpr_mean, tpr_std = [], []

                        precision_mean, precision_std = [], []
                        recall_mean, recall_std = [], []

                        fpr_of_means = []
                        tpr_of_means = []
                        precision_of_means = []
                        recall_of_means = []

                        fpr_single_runs = [[] for _ in range(len(tracing_stats[thresholds[0]][policy]['isolate']['tn']))]
                        tpr_single_runs = [[] for _ in range(len(tracing_stats[thresholds[0]][policy]['isolate']['tn']))]

                        if p_adoption is not None:
                            if name == 'PanCast':
                                p_tracing = get_tracing_probability('PanCast',
                                                                            p_adoption=p_adoption,
                                                                            p_manual_reachability=p_manual_reachability,
                                                                            p_recall=p_recall,
                                                                            p_beacon=p_beac)
                                p_tracings.append(p_tracing)
                            elif name == 'SPECT':
                                p_tracing = get_tracing_probability('SPECTS',
                                                                          p_adoption=p_adoption,
                                                                          p_manual_reachability=p_manual_reachability,
                                                                          p_recall=p_recall,
                                                                          p_beacon=None)
                                p_tracings.append(p_tracing)
                            else:
                                raise NotImplementedError()

                        for t, thres in enumerate(thresholds):

                            stats = copy.copy(tracing_stats[thres][policy][action])

                            if p_adoption is not None:
                                orig_fp = np.asarray(stats['fp'])
                                orig_tp = np.asarray(stats['tp'])
                                stats['fp'] = np.asarray(stats['fp']) * p_tracing
                                stats['tp'] = np.asarray(stats['tp']) * p_tracing
                                stats['tn'] = np.asarray(stats['tn']) + orig_fp * (1-p_tracing)
                                stats['fn'] = np.asarray(stats['fn']) + orig_tp * (1-p_tracing)

                            # FPR = FP/(FP + TN) [isolate + not infected / not infected]
                            # [if FP = 0 and TN = 0, set to 0]
                            fprs = stats['fp'] / (stats['fp'] + stats['tn'])
                            fprs = np.nan_to_num(fprs, nan=0.0)
                            fpr_mean.append(np.mean(fprs).item())
                            fpr_std.append(np.std(fprs).item())
                            fpr_of_means.append(stats['fp'].mean() / (stats['fp'].mean() + stats['tn'].mean()))

                            for r in range(len(fpr_single_runs)):
                                fpr_single_runs[r].append(fprs[r])

                            # TPR = TP/(TP + FN) [isolate + infected / infected]
                            # = RECALL
                            # [if TP = 0 and FN = 0, set to 0]
                            tprs = stats['tp'] / (stats['tp'] + stats['fn'])
                            tprs = np.nan_to_num(tprs, nan=0.0)
                            tpr_mean.append(np.mean(tprs).item())
                            tpr_std.append(np.std(tprs).item())
                            tpr_of_means.append(stats['tp'].mean() / (stats['tp'].mean() + stats['fn'].mean()))

                            for r in range(len(tpr_single_runs)):
                                tpr_single_runs[r].append(tprs[r])

                            # precision = TP/(TP + FP)
                            precs = stats['tp'] / (stats['tp'] + stats['fp'])
                            precs = np.nan_to_num(precs, nan=0.0)
                            precision_mean.append(np.mean(precs).item())
                            precision_std.append(np.std(precs).item())
                            precision_of_means.append(stats['tp'].mean() / (stats['tp'].mean() + stats['fp'].mean()))

                            # if i == 0:
                            if verbose:
                                print("{:1.3f}   TP {:5.2f} FP {:5.2f}  TN {:5.2f}  FN {:5.2f}".format(
                                    thres, stats['tp'].mean(), stats['fp'].mean(), stats['tn'].mean(), stats['fn'].mean()
                                ))
                                if t == len(thresholds) - 1:
                                    print(" P {:5.2f}  N {:5.2f}".format(
                                        (stats['fn'] + stats['tp']).mean(), (stats['fp'] + stats['tn']).mean()
                                    ))

                        policies[name] = {'fpr': fpr_of_means,
                                          'tpr': tpr_of_means,
                                          'prec': precision_of_means}

                        if name == 'SPECT':
                            name += 'S'

                        # lines
                        if p_adoption is not None:
                            colors = ['#377eb8', '#bd0026', '#f03b20', '#fd8d3c', '#fecc5c', '#ffffb2']
                            if name == 'PanCast':
                                longname = name + f', {int(p_beac*100)}\%'
                            else:
                                longname = name

                            axs[plt_index].plot(fpr_of_means, tpr_of_means, linestyle='-', label=longname, c=colors[colorind])
                            colorind += 1
                        else:
                            colors = ['#377eb8', '#e41a1c',]
                            axs[0].plot(fpr_of_means, tpr_of_means, linestyle='-', label=name, c=colors[j])

                        # axs[1].plot(tpr_of_means, precision_of_means, linestyle='-', label=name, c=self.color_different_scenarios[j])
                        # axs[0].plot(fpr_mean, tpr_mean, linestyle='-', label=name, c=self.color_different_scenarios[j])
                        # axs[1].plot(tpr_mean, precision_mean, linestyle='-', label=name, c=self.color_different_scenarios[j])

                # for each FPR bucket, collect TPR and prec values
                policy_bin_values = dict()
                n_bins = 6
                bins = np.linspace(0.0, 1.0, n_bins)
                for n in range(bins.shape[0] - 1):
                    print(f'index {n + 1} : {bins[n]} - {bins[n + 1]}')

                for policy in ['SPECT', 'PanCast']:
                    fprs = np.array(policies[policy]['fpr'])
                    tprs = np.array(policies[policy]['tpr'])
                    precs = np.array(policies[policy]['prec'])

                    inds = np.digitize(fprs, bins)

                    bin_values_fpr = collections.defaultdict(list)
                    bin_values_tpr = collections.defaultdict(list)
                    bin_values_prec = collections.defaultdict(list)

                    for i in range(fprs.shape[0]):
                        bin_values_fpr[inds[i]].append(fprs[i])
                        bin_values_tpr[inds[i]].append(tprs[i])
                        bin_values_prec[inds[i]].append(precs[i])

                    # form mean of each bucket
                    policy_bin_values[policy] = {
                        'fpr' : {k:np.array(lst).mean().item() for k, lst in bin_values_fpr.items()},
                        'tpr' : {k:np.array(lst).mean().item() for k, lst in bin_values_tpr.items()},
                        'prec': {k: np.array(lst).mean().item() for k, lst in bin_values_prec.items()},
                    }

                # print improvement pancast over spect
                # pprint.pprint(policy_bin_values)
                for metric in ['tpr', 'prec']:

                    relative_percentage = []

                    for ind in policy_bin_values['SPECT']['fpr'].keys():

                        # only check bins where both have values
                        if (ind not in policy_bin_values['PanCast'][metric].keys()) or\
                           (ind not in policy_bin_values['SPECT'][metric].keys()):
                           continue

                        # ignore edge bins
                        if ind <= 1 or ind >= n_bins - 1:
                            continue

                        relative_percentage.append(
                            (ind, policy_bin_values['PanCast'][metric][ind] / policy_bin_values['SPECT'][metric][ind])
                        )

                    try:
                        argmaxval, maxval = max(relative_percentage, key=lambda x: x[1])
                        print('Maximum relative % PanCast/SPECT (excluding boundary)', metric, maxval * 100, 'bin: ', argmaxval)
                    except ValueError:
                        print('Could not compute Maximum relative % PanCast/SPECT (excluding boundary)')

            maximum = 1.0 if p_adoption is None else max(p_tracings)
            # for ax in axs:
            axs[plt_index].set_xlim((0.0, maximum))
            axs[plt_index].set_ylim((0.0, maximum))

            # diagonal
            xs = np.linspace(0, maximum, num=500)
            axs[plt_index].plot(xs, xs, linestyle='dotted', c='black')

            if use_medical_labels:
                axs[plt_index].set_xlabel('1-Specificity')
                # axs[plt_index].set_ylabel('Sensitivity')
                axs[0].set_ylabel('Sensitivity')
            else:
                axs[plt_index].set_xlabel('FPR')
                # axs[plt_index].set_ylabel('TPR')
                axs[0].set_ylabel('TPR')

            # axs[1].set_xlabel('Recall')
            # axs[1].set_ylabel('Precision')

            # Set default axes style
            # self._set_default_axis_settings(ax=ax)
            axs[plt_index].set_aspect('equal', 'box')
            axs[plt_index].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if p_adoption:
                axs[plt_index].set_title(f'{int(p_adoption*100)}\% adoption')
            # else:
            #     leg = axs[plt_index].legend(loc='lower right')
        # leg = axs[1].legend(loc='top right')

        if p_adoption is not None:
            # axs[0].legend(loc='lower left', bbox_to_anchor=(-1.5, 0.18),
            #           borderaxespad=0, frameon=True)
            axs[-1].legend(loc='lower left', bbox_to_anchor=(1.1, 0.18),
                          borderaxespad=0, frameon=True)
        else:
            leg = axs[0].legend(loc='lower right')
        # subplot_adjust = subplot_adjust or {'bottom':0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        # plt.subplots_adjust(**subplot_adjust)

        plt.tight_layout()
        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        # plt.tight_layout()
        if NO_PLOT:
            plt.close()
        return

    def reff_heatmap(self, xlabel, ylabel, paths, path_labels, figformat='double',
                     filename='reff_heatmap_0', figsize=None, acc=500, 
                     relative_window=(0.25, 0.75)):
        ''''
        Plots heatmap of average R_t
            paths:              list with tuples (x, y, path)
            relative_window:    relative range of max_time used for R_t average
        '''
        # set double figure format
        self._set_matplotlib_params(format=figformat)

        # draw figure
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # extract data
        reff_means_all = []
        for p in paths:

            reff_means = []
            for xval, yval, path in p:

                data = load_condensed_summary(path, acc)
                rtdata = data['nbinom_dist']
                n_rollouts = data['metadata'].random_repeats
                max_time = data['max_time']

                l_max_time = relative_window[0] * max_time 
                r_max_time = relative_window[1] * max_time

                # filter time window
                rtdata_window = rtdata.loc[(l_max_time <= rtdata['t0']) & (rtdata['t1'] <= r_max_time)]
                reff_mean = np.mean(rtdata_window["Rt"])
        
                reff_means.append((xval, yval, reff_mean))

            reff_means_all.append(reff_means)

        # find min and max for both plots
        zmins, zmaxs = [], []
        for reff_means in reff_means_all:
            x, y, z = zip(*reff_means)
            zmins.append(min(z))
            zmaxs.append(max(z))

        zmin, zmax = min(zmins), max(zmaxs)

        # generate heatmaps
        for t, title in enumerate(path_labels):
            
            x, y, z = zip(*reff_means_all[t])
            xbounds = min(x), max(x)
            ybounds = min(y), max(y)

            # contour interpolation
            xi = np.linspace(xbounds[0], xbounds[1], 100)
            yi = np.linspace(ybounds[0], ybounds[1], 100)
            zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

            n_bins = 15
            axs[t].contour(xi, yi, zi, n_bins, linewidths=0.5, colors='k', norm=colors.Normalize(vmin=zmin, vmax=zmax))

            if t == 0:
                contourplot = axs[t].contourf(xi, yi, zi, n_bins, cmap=plt.cm.jet, norm=colors.Normalize(vmin=zmin, vmax=zmax))
            else:
                _ = axs[t].contourf(xi, yi, zi, n_bins, cmap=plt.cm.jet, norm=colors.Normalize(vmin=zmin, vmax=zmax))

            axs[t].set_xlabel(xlabel)
            if t == 0:
                axs[t].set_ylabel(ylabel)
            axs[t].set_xlim(xbounds)
            axs[t].set_ylim(ybounds)
            axs[t].set_title(title)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(contourplot, cax=cbar_ax)


        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return
    
    def relative_quantity_heatmap(self, mode, xlabel, ylabel, paths, path_labels, baseline_path, zmax=None, figformat='double',
                     filename='reff_heatmap_0', figsize=None, acc=500, interpolate='linear', # or `cubic`
                     width_ratio=4, cmap='jet'):
        ''''
        Plots heatmap of average R_t
            paths:              list with tuples (x, y, path)
            relative_window:    relative range of max_time used for R_t average
        '''
        if mode == 'cumu_infected':
            key = 'cumu_infected_'
            colorbar_label = r'\% reduction of infections'
        elif mode == 'hosp':
            key = 'hosp_'
            colorbar_label = r'\% reduction of peak hosp.'
        elif mode == 'dead':
            key = 'cumu_dead_'
            colorbar_label = r'\% reduction of deaths'

        # set double figure format
        self._set_matplotlib_params(format=figformat)

        # draw figure
        fig, axs = plt.subplots(1, 2, figsize=figsize,  gridspec_kw={'width_ratios': [1, width_ratio]})

        baseline_data = load_condensed_summary(baseline_path)
        baseline_series = baseline_data[key + 'mu']

        # extract data
        zval_means_all = []
        for p in paths:

            zval_means = []
            for xval, yval, path in p:
                data = load_condensed_summary(path, acc)

                # extract z value given (x, y)
                series = data[key + 'mu']
                if 'cumu' in key:
                    # last
                    zval = (1 - series[-1] / baseline_series[-1]) * 100
                else:
                    # peak
                    zval = (1 - series.max() / baseline_series.max()) * 100

                zval_means.append(((xval * 100 if xval is not None else xval), yval * 100, zval.item()))

            zval_means_all.append(zval_means)

        # define min and max for both plots
        if zmax:
            zmin, zmax_color, zmax_colorbar = 0, zmax, zmax
        else:
            zmin, zmax_color, zmax_colorbar = 0, 90, 90
        stepsize = 5
        norm = colors.Normalize(vmin=zmin, vmax=zmax_color)
        levels = np.arange(zmin, zmax_colorbar + stepsize, stepsize)

        # generate heatmaps
        for t, title in enumerate(path_labels):

            x, y, z = zip(*zval_means_all[t])

            if x[0] is None:
                # move 1D data on a 2D manifold for plotting
                xbounds = (-0.1, 0.1)
                ybounds = min(y), max(y)

                x = [xbounds[0] for _ in y] + [xbounds[1] for _ in y]
                y = y + y
                z = z + z

                axs[t].xaxis.set_major_formatter(plt.NullFormatter())
                axs[t].xaxis.set_minor_formatter(plt.NullFormatter())
                axs[t].xaxis.set_major_locator(plt.NullLocator())
                axs[t].xaxis.set_minor_locator(plt.NullLocator())

            else:
                x = np.log(x)
                xbounds = min(x), max(x)
                ybounds = min(y), max(y)

                axs[t].set_xlabel(xlabel)

                # x ticks
                @ticker.FuncFormatter
                def major_formatter(x_, pos):
                    return r"{:3.0f}".format(np.exp(x_))

                # for some reason, FixedLocator makes tick labels falsely bold
                # axs[t].xaxis.set_major_locator(ticker.FixedLocator(x))
                axs[t].xaxis.set_major_locator(CustomSitesProportionFixedLocator())
                axs[t].xaxis.set_major_formatter(major_formatter)

            # contour interpolation
            xi = np.linspace(xbounds[0], xbounds[1], 100)
            yi = np.linspace(ybounds[0], ybounds[1], 100)
            zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method=interpolate)

            # contour plot
            axs[t].contour(xi, yi, zi, linewidths=0.5, colors='k', norm=norm, levels=levels)
            contourplot = axs[t].contourf(xi, yi, zi, cmap=cmap, norm=norm, levels=levels)

            # axis
            axs[t].set_xlim(xbounds)
            axs[t].set_ylim(ybounds)
            axs[t].set_title(title)

            axs[t].set_yticks(list(axs[t].get_yticks())[1:] + [ybounds[0]])
            axs[t].set_yticks([10, 25, 40, 55, 70, 85, 100])

            if t == 0:
                axs[t].set_ylabel(ylabel)
            else:
                pass

        # layout and color bar
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)

        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.87, 0.17, 0.05, 0.7])
        cbar = matplotlib.colorbar.ColorbarBase(
            cbar_ax, cmap=plt.cm.RdYlGn,
            norm=norm,
            boundaries=levels,
            ticks=levels[::2],
            orientation='vertical')
        cbar.set_label(colorbar_label, labelpad=5.0)

        # save
        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def manual_tracing_heatmap(self, mode, xlabel, ylabel, paths, path_labels, baseline_path, figformat='double',
                                  filename='manual_tracing_heatmap_0', figsize=None, acc=500, interpolate='linear',  # or `cubic`
                                  width_ratio=4, cmap='jet'):
        ''''
        Plots heatmap of average R_t
            paths:              list with tuples (x, y, path)
            relative_window:    relative range of max_time used for R_t average
        '''
        if mode == 'cumu_infected':
            key = 'cumu_infected_'
            colorbar_label = r'\% reduction of infections'
        elif mode == 'hosp':
            key = 'hosp_'
            colorbar_label = r'\% reduction of peak hosp.'
        elif mode == 'dead':
            key = 'cumu_dead_'
            colorbar_label = r'\% reduction of deaths'

        # set double figure format
        self._set_matplotlib_params(format=figformat)

        # draw figure
        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, width_ratio]})


        # extract data
        zval_means_all = []
        for p in paths:

            zval_means = []
            for xval, yval, path in p:
                data = load_condensed_summary(path, acc)

                # extract z value given (x, y)
                series = data[key + 'mu']
                if 'cumu' in key:
                    # last
                    zval = (1 - series[-1] / baseline_series[-1]) * 100
                else:
                    # peak
                    zval = (1 - series.max() / baseline_series.max()) * 100

                zval_means.append(((xval * 100 if xval is not None else xval), yval * 100, zval.item()))

            zval_means_all.append(zval_means)

        # define min and max for both plots
        zmin, zmax_color, zmax_colorbar = 0, 90, 90
        stepsize = 5
        norm = colors.Normalize(vmin=zmin, vmax=zmax_color)
        levels = np.arange(zmin, zmax_colorbar + stepsize, stepsize)

        # generate heatmaps
        for t, title in enumerate(path_labels):

            x, y, z = zip(*zval_means_all[t])

            if x[0] is None:
                # move 1D data on a 2D manifold for plotting
                xbounds = (-0.1, 0.1)
                ybounds = min(y), max(y)

                x = [xbounds[0] for _ in y] + [xbounds[1] for _ in y]
                y = y + y
                z = z + z

                axs[t].xaxis.set_major_formatter(plt.NullFormatter())
                axs[t].xaxis.set_minor_formatter(plt.NullFormatter())
                axs[t].xaxis.set_major_locator(plt.NullLocator())
                axs[t].xaxis.set_minor_locator(plt.NullLocator())

            else:
                x = np.log(x)
                xbounds = min(x), max(x)
                ybounds = min(y), max(y)

                axs[t].set_xlabel(xlabel)

                # x ticks
                @ticker.FuncFormatter
                def major_formatter(x_, pos):
                    return r"{:3.0f}".format(np.exp(x_))

                # for some reason, FixedLocator makes tick labels falsely bold
                # axs[t].xaxis.set_major_locator(ticker.FixedLocator(x))
                axs[t].xaxis.set_major_locator(CustomSitesProportionFixedLocator())
                axs[t].xaxis.set_major_formatter(major_formatter)

            # contour interpolation
            xi = np.linspace(xbounds[0], xbounds[1], 100)
            yi = np.linspace(ybounds[0], ybounds[1], 100)
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method=interpolate)

            # contour plot
            axs[t].contour(xi, yi, zi, linewidths=0.5, colors='k', norm=norm, levels=levels)
            contourplot = axs[t].contourf(xi, yi, zi, cmap=cmap, norm=norm, levels=levels)

            # axis
            axs[t].set_xlim(xbounds)
            axs[t].set_ylim(ybounds)
            axs[t].set_title(title)

            axs[t].set_yticks(list(axs[t].get_yticks())[1:] + [ybounds[0]])
            axs[t].set_yticks([10, 25, 40, 55, 70, 85, 100])

            if t == 0:
                axs[t].set_ylabel(ylabel)
            else:
                pass

        # layout and color bar
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)

        # [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.87, 0.17, 0.05, 0.7])
        cbar = matplotlib.colorbar.ColorbarBase(
            cbar_ax, cmap=plt.cm.RdYlGn,
            norm=norm,
            boundaries=levels,
            ticks=levels[::2],
            orientation='vertical')
        cbar.set_label(colorbar_label, labelpad=5.0)

        # save
        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def compare_peak_reduction(self, path_list, baseline_path=None, ps_adoption=None, labels=None, title=None,
                               mode='cumu_infected', show_reduction=True, log_xscale=True, log_yscale=False, ylim=None,
                               area_population=None, colors=None, show_baseline=True, combine_summaries=False,
                               show_significance=None, sig_options=None,
                               figformat='double', filename='cumulative_reduction', figsize=None,
                               show_legend=True, legend_is_left=False, subplot_adjust=None, box_plot=False):

        show_reduction = show_reduction and (baseline_path is not None)

        if ylim is None:
            ylim = (0, 100)

        if mode == 'cumu_infected':
            key = 'cumu_infected_'
            ylabel = r'\% reduction of infections' if show_reduction else 'Cumulative infected'
        elif mode == 'hosp':
            key = 'hosp_'
            ylabel = r'\% reduction of peak hosp.' if show_reduction else 'Peak hospitalizations'
        elif mode == 'dead':
            key = 'cumu_dead_'
            ylabel = r'\% reduction of fatalities' if show_reduction else 'Fatalities'
        elif mode == 'r_eff':
            ylabel = r'\% reduction of $R_{\textrm{eff}}$' if show_reduction else r'$R_{\textrm{eff}}$'
            assert area_population is not None, 'Requires argument area_population for R_eff plots'

        # Set double figure format
        self._set_matplotlib_params(format=figformat)
        # Draw figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ps_adoption = np.asarray(ps_adoption) * 100

        # colors = self.color_different_scenarios
        if colors is None:
            colors = ['#377eb8', '#bd0026', '#f03b20', '#fd8d3c', '#fecc5c', '#ffffb2']
            #colors = [ '#bd0026', '#f03b20', '#fd8d3c', '#fecc5c', '#377eb8',]
        zorders = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        if mode == 'r_eff':
            baseline_data = load_condensed_summary(baseline_path)
            baseline_mu, baseline_std = get_rt(baseline_data, p_infected=0.1, area_population=area_population,
                                               average_up_to_p_infected=True)
            if show_baseline:
                bars = ax.errorbar(['0'], [baseline_mu], yerr=[baseline_std], label=labels[-1],
                                   c='#31a354', marker="o", linestyle="none")
        else:
            baseline_mu, baseline_sig = get_peak_mu_and_std(path=baseline_path, key=key, combine_summaries=combine_summaries)

        means = []
        stds = []
        for i, paths in enumerate(path_list):
            rel_mean = []
            rel_std = []

            if mode == 'r_eff':
                for path in paths:
                    data = load_condensed_summary(path)
                    rt, rt_std = get_rt(data, p_infected=0.1, area_population=area_population,
                                        average_up_to_p_infected=True)
                    rel_mean.append(rt)
                    rel_std.append(rt_std)
            else:
                for path in paths:
                    mu, sig = get_peak_mu_and_std(path=path, key=key, combine_summaries=combine_summaries)
                    rel_mean.append(mu)
                    rel_std.append(sig)

            if show_reduction:
                rel_mean = (1 - np.asarray(rel_mean) / baseline_mu) * 100
                rel_std = np.asarray(rel_std) / baseline_mu * 100

            means.append(rel_mean)
            stds.append(rel_std)

            if not box_plot:
                bars = ax.errorbar(ps_adoption, rel_mean, yerr=rel_std, label=labels[i],
                               c=colors[i], linestyle='-', elinewidth=0.8, capsize=3.0, zorder=zorders[i])
            else:
                ps_adoption_string = [str(int(element)) for element in ps_adoption]
                offset = (len(path_list))/2 * 0.11
                trans = ax.transData + ScaledTranslation(-offset + (i+1/2)/len(path_list) * 2 * offset, 0, fig.dpi_scale_trans)
                bars = ax.errorbar(ps_adoption_string, rel_mean, yerr=rel_std, label=labels[i],
                                   c=colors[i], marker="o", linestyle="none", transform=trans)

        if log_yscale:
            ax.set_yscale('log')
            ax.set_yticks([ylim[0]]+list(ps_adoption))
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        if mode == 'r_eff':
            ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

        ax.set_ylim(ymax=ylim[1], ymin=ylim[0])

        if not box_plot:
            if log_xscale:
                ax.set_xscale('log')
            ax.set_xlim(left=np.min(ps_adoption), right=104)
            ax.set_ylim(ymax=ylim[1], ymin=ylim[0])
            ax.set_xticks(ps_adoption)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        else:
            ax.margins(x=0.15)

        ax.set_ylabel(ylabel)
        ax.set_xlabel('\% adoption')

        if title:
            ax.set_title(title)

        if show_significance is not None:
            sig_options_updated = {'lhs_xshift': 0.013,
                        'rhs_xshift': 0.13,
                        'height': 1.0,
                        'text_offset': 0.0,
                        'same_height': True}
            if isinstance(sig_options, dict):
                for key, value in sig_options.items():
                    sig_options_updated[key] = value
            sig_options = sig_options_updated

            if sig_options['same_height']:
                ymax = np.max(np.asarray(means) + np.asarray(stds), axis=0)
                if log_yscale:
                    ys = np.exp(np.log(ymax) + np.log((ylim[1] - ylim[0]) * 0.008))
                else:
                    ys = ymax + (ylim[1] - ylim[0]) * 0.008
                print(np.shape(ys))
                ys = np.repeat(np.expand_dims(ys, axis=0), len(path_list), axis=0)
                print(np.shape(ys))
            else:
                ys = np.asarray(means) + np.asarray(stds)

            for j in range(len(ps_adoption_string)):
                #y = ys[j]

                for i in range(len(path_list) - 1):
                    if show_significance == 'no_bars':
                        mean1 = means[0][j]
                        std1 = stds[0][j]
                    else:
                        mean1 = means[i][j]
                        std1 = stds[i][j]
                    mean2 = means[i+1][j]
                    std2 = stds[i+1][j]

                    rollouts = 400 if combine_summaries else 100
                    is_significant, p_value = independent_ttest(mean1, std1, mean2, std2, rollouts=rollouts, alpha=0.05)
                    print('p-value: ', p_value)
                    y = ys[i+1][j]
                    x1 = j + sig_options['lhs_xshift']
                    x2 = j + sig_options['rhs_xshift']
                    h = sig_options['height']
                    if log_yscale:
                        h = np.exp(np.log(y) + np.log(h))
                    if show_significance == 'no_bars':
                        i += 1
                    offset = (len(path_list)) / 2 * 0.11
                    trans = ax.transData + ScaledTranslation(-offset + sig_options['text_offset'] + (i + 1 / 2) / len(path_list) * 2 * offset, 0,
                                                             fig.dpi_scale_trans)

                    if show_significance == 'no_bars':
                        if is_significant:
                            plt.text(j, y + h, "*", ha='center', va='bottom', color='black', transform=trans)
                    elif show_significance == 'bars':
                        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color='black', transform=trans)
                        if is_significant:
                            plt.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color='black', transform=trans)

        if show_legend:
                # legend
                if legend_is_left:
                    leg = ax.legend(loc='upper left',
                                    bbox_to_anchor=(0.001, 0.999),
                                    bbox_transform=ax.transAxes,
                                    )
                else:
                    leg = ax.legend(loc='upper right',
                                    bbox_to_anchor=(0.999, 0.999),
                                    bbox_transform=ax.transAxes,
                                    )
                if legend_is_left == 'outside':
                    leg = ax.legend(loc='lower left',
                                    bbox_to_anchor=(0.001,0.001),
                                    bbox_transform=ax.transAxes,
                                    )

        subplot_adjust = subplot_adjust or {'bottom': 0.14, 'top': 0.98, 'left': 0.12, 'right': 0.96}
        plt.subplots_adjust(**subplot_adjust)

        plt.savefig('plots/' + filename + '.pdf', format='pdf', facecolor=None,
                dpi=DPI, bbox_inches='tight')

        if NO_PLOT:
            plt.close()
        return

    def beta_parameter_heatmap(self, country, area, calibration_state, G_is_objective=True, estimate_mobility_reduction=False,
                               figsize=(3, 3), cmap='viridis_r', levels=None, scatter=False, ceil=None, xmin=None,
                               xmax=None, ymin=None, ymax=None):

        self._set_matplotlib_params(format='neurips-double')

        param_bounds = calibration_model_param_bounds_single
        sim_bounds = pdict_to_parr(
            pdict=param_bounds,
            multi_beta_calibration=False,
            estimate_mobility_reduction=estimate_mobility_reduction,
        ).T

        state = load_state(calibration_state)
        train_theta = state['train_theta']
        train_G = state['train_G']

        mob_settings = calibration_mob_paths[country][area][1]
        with open(mob_settings, 'rb') as fp:
            mob_kwargs = pickle.load(fp)

        data_start_date = calibration_start_dates[country][area]
        data_end_date = calibration_lockdown_dates[country]['end']

        unscaled_area_cases = collect_data_from_df(country=country, area=area, datatype='new',
                                                start_date_string=data_start_date, end_date_string=data_end_date)
        assert (len(unscaled_area_cases.shape) == 2)

        # Scale down cases based on number of people in town and region
        sim_cases = downsample_cases(unscaled_area_cases, mob_kwargs)
        n_days, n_age = sim_cases.shape

        G_obs = torch.tensor(sim_cases).reshape(1, n_days * n_age)
        G_obs_aggregate = torch.tensor(sim_cases).sum(dim=-1)

        def objective(G):
            return - (G - G_obs_aggregate).pow(2).sum(dim=-1) / n_days

        bo_result = np.zeros((train_theta.shape[0] + 4, 3))
        for t in range(train_theta.shape[0]):
            theta = train_theta[t]
            G = train_G[t]
            real_theta = transforms.unnormalize(theta, sim_bounds)
            obj = G.item() if G_is_objective else objective(G).item()

            bo_result[t, 0:2] = real_theta
            bo_result[t, 2] = - obj

        # fill corners with closest points

        for t, (x_coord, y_coord) in enumerate([
            (sim_bounds[0, 0], sim_bounds[0, 0]),
            (sim_bounds[0, 0], sim_bounds[1, 0]),
            (sim_bounds[1, 0], sim_bounds[0, 1]),
            (sim_bounds[1, 0], sim_bounds[1, 1]),
        ]):
            idx = t - 4 # fill last 4 rows

            # find closest point
            dist_to_pt = (x_coord - bo_result[:-4, 0]) ** 2 + (y_coord - bo_result[:-4, 1]) ** 2
            argmin = np.argmin(dist_to_pt)

            # fill corner with closest known objective
            bo_result[idx, 0] = x_coord
            bo_result[idx, 1] = y_coord
            bo_result[idx, 2] = bo_result[argmin, 2]

        
        # ceil
        if ceil is not None:
            bo_result[:, 2] = np.minimum(bo_result[:, 2], ceil)

        # plot
        dim_names = {
            0: r'$\beta$',
            1: r'$\xi$',
        }

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.subplots_adjust(top=0.8)

        x = bo_result[:, 0]
        y = bo_result[:, 1]

        # z = bo_result[:, 2]
        # z = np.sqrt(bo_result[:, 2])
        z = np.log(bo_result[:, 2])
        # z = np.log10(bo_result[:, 2])
        # z = np.log(np.sqrt(bo_result[:, 2]))

        # contour interpolation
        xi = np.linspace(xmin or sim_bounds[0, 0], xmax or sim_bounds[1, 0], 100)
        yi = np.linspace(ymin or sim_bounds[0, 1], ymax or sim_bounds[1, 1], 100)
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

        if levels is not None:
            ax.contour(xi, yi, zi, levels, linewidths=0.5, colors='k')
            contourplot = ax.contourf(xi, yi, zi, levels, cmap=cmap)
        else:
            ax.contour(xi, yi, zi, linewidths=0.5, colors='k')
            contourplot = ax.contourf(xi, yi, zi, cmap=cmap)

        if scatter:
            np.set_printoptions(precision=4, suppress=True)
            top_points = np.where((bo_result[:, 2] <= 1000000) & (np.arange(bo_result.shape[0]) >= 10))
            print('BO evaluations')

            print(bo_result[top_points])
            ax.scatter(bo_result[top_points, 0], bo_result[top_points, 1], color='blue', s=5)

        # colorbar
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(contourplot, cax=cbar_ax)
        fig.colorbar(contourplot)

        # axis
        ax.set_xlabel(dim_names[0], labelpad=2)
        ax.set_ylabel(dim_names[1], labelpad=5)
        ax.set_xlim((xmin or (sim_bounds[0, 0] + 0.001),
                     xmax or (sim_bounds[1, 0] - 0.001)))
        ax.set_ylim((ymin or (sim_bounds[0, 1] + 0.001),
                     ymax or (sim_bounds[1, 1] - 0.001)))

        ax.set_title(r'$\log f(\theta)$', y=1.0)

        plt.tight_layout()
        plt.savefig(f'plots/bo-result-{country}-{area}.pdf', format='pdf', facecolor=None, dpi=DPI, bbox_inches='tight')

        plt.show()


def combine_multiple_summaries(paths, key):
    assert isinstance(paths, list)
    mus = []
    stds = []
    for path in paths:
        data = load_condensed_summary(path)
        maxidx = np.argmax(data[key + 'mu'])
        mus.append(data[key + 'mu'][maxidx])
        stds.append(data[key + 'sig'][maxidx])
    mu = np.mean(mus)
    std = np.sqrt(np.mean(np.square(np.asarray(stds))))
    return mu, std


def get_peak_mu_and_std(path, key, combine_summaries=False):
    if isinstance(path, list):
        assert combine_summaries
        mu, sig = combine_multiple_summaries(paths=path, key=key)
    else:
        data = load_condensed_summary(path)
        maxidx = np.argmax(data[key + 'mu'])
        mu = data[key + 'mu'][maxidx]
        sig = data[key + 'sig'][maxidx]
    return mu, sig


def independent_ttest(mean1, std1, mean2, std2, rollouts, alpha):
    t_stat = (mean2 - mean1) / np.sqrt(std1**2/np.sqrt(rollouts) + std2**2/np.sqrt(rollouts))
    df = 2 * rollouts - 2
    p = (1.0 - scipy.stats.t.cdf(abs(t_stat), df)) * 2.0
    return p < alpha, p


