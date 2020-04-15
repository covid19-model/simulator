import time
import bisect
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
import scipy as sp
import os
import matplotlib.pyplot as plt
import random

from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours

import asyncio
import threading

import json
import tornado.ioloop
import tornado.httpserver
from tornado.web import RequestHandler
import requests

from lib.priorityqueue import PriorityQueue
from lib.dynamics import DiseaseModel
from lib.mobilitysim import MobilitySimulator
from bayes_opt import BayesianOptimization
from lib.parallel import *

SIMPLIFIED_OPT = True

def format_opt_to_sim(opt_params, n_betas):
    '''
    Convert bayes_opt parameter format into our format
    '''

    if SIMPLIFIED_OPT:
        return {
            'betas' : [opt_params['beta'] for _ in range(n_betas)],
            'alpha': opt_params['alpha'], 
            'mu': opt_params['mu']
        }

    else:
        sim_params = {
            'betas' : [None for _ in range(n_betas)],
            'alpha': None, 
            'mu': None
        }
        for k, v, in opt_params.items():
            if 'betas' in k:
                sim_params['betas'][int(k[5:])] = v
            else:
                sim_params[k] = v
        return sim_params


def format_sim_to_opt(sim_params):
    '''
    Convert our format into bayes opt format
    '''
    if SIMPLIFIED_OPT:
        return {
            'beta' : sim_params['betas'][0],
            'alpha': sim_params['alpha'], 
            'mu': opt_params['mu']
        }
        
    else:
        opt_params = {'betas' + str(i) : p for i, p in enumerate(sim_params['betas'])}
        opt_params.update({
            'alpha': sim_params['alpha'], 
            'mu': sim_params['mu']
        })
        return opt_params

def convert_timings_to_daily(timings, time_horizon):
    '''

    Converts batch of size N of timings of M individuals in a time horizon 
    of `time_horizon` in hours into daily aggregate cases

    Argument:
        timings :   np.array of shape (N, M)

    Argument:
        timings :   np.array of shape (N, T / 24)
    '''
    if len(timings.shape) == 1:
        timings = np.expand_dims(timings, axis=0)

    arr = np.array([
        np.sum((timings >= t * 24) &
               (timings < (t + 1) * 24), axis=1)
        for t in range(0, int(time_horizon // 24))]).T

    return arr


def convert_timings_to_cumulative_daily(timings, time_horizon):
    '''

    Converts batch of size N of timings of M individuals in a time horizon 
    of `time_horizon` in hours into daily cumulative aggregate cases 

    Argument:
        timings :   np.array of shape (N, M)

    Argument:
        timings :   np.array of shape (N, T / 24)
    '''
    if len(timings.shape) == 1:
        timings = np.expand_dims(timings, axis=0)

    cumulative = np.array([
        np.sum((timings < (t + 1) * 24), axis=1)
        for t in range(0, int(time_horizon // 24))]).T

    return cumulative


def loss_daily(predicted_confirmed_times, targets_daily, time_horizon, power=2.0):
    '''
    Daily loss: 
        total squared error between average predicted daily cases and true daily cases
    '''
    # predicted_confirmed_daily = convert_timings_to_daily(predicted_confirmed_times, time_horizon)
    predicted_confirmed_daily = convert_timings_to_cumulative_daily(predicted_confirmed_times, time_horizon)
    ave_predicted_confirmed_daily = predicted_confirmed_daily.mean(axis=0)
    loss = np.power(np.abs(ave_predicted_confirmed_daily - targets_daily), power).mean()
    return loss


def multimodal_loss_daily(preds, weights, targets, time_horizon, power=2.0):
    '''
    Multimodal Daily loss: 
        Same as loss_daily but considering several weighted metrics (e.g. positive, recovered, deceased)
    '''
    loss = 0
    for w, pred, target in zip(weights, preds, targets):
        # pred = convert_timings_to_daily(pred, time_horizon)
        pred = convert_timings_to_cumulative_daily(pred, time_horizon)
        ave_pred = pred.mean(axis=0)
        loss += w * np.power(np.abs(ave_pred - target), power).mean()
    return loss


def make_loss_function(mob_settings, distributions, targets, time_horizon, param_bounds,
    initial_seeds, testing_params, random_repeats, num_site_types,
    cpu_count, measure_list, loss, num_people, site_loc, home_loc, c, extra_params=None):
    

    '''
    Returns function executable by optimizer with desired loss
    '''

    with open(f'logger_{c}.txt', 'w+') as logfile:
        logfile.write(f'Log run: seed = {c}\n\n')

    def f(opt_params):

        # convert bayes_opt parameter format into our format
        sim_params = format_opt_to_sim(opt_params, n_betas=num_site_types)
         
        # launch in parallel
        summary = launch_parallel_simulations(
            mob_settings=mob_settings,
            distributions=distributions,
            random_repeats=random_repeats,
            cpu_count=cpu_count,
            params=sim_params,
            initial_seeds=initial_seeds,
            testing_params=testing_params,
            measure_list=measure_list,
            max_time=time_horizon,
            num_people=num_people,
            site_loc=site_loc,
            home_loc=home_loc,
            verbose=False)

        if loss == 'loss_daily':
            return summary.state_started_at['posi']
        elif loss == 'multimodal_loss_daily':
            return (summary.state_started_at['posi'], summary.state_started_at['resi'],  summary.state_started_at['dead'])
        else:
            raise ValueError('Unknown loss function')


    if loss == 'loss_daily':

        def loss_function(**kwargv):
            predicted_confirmed_times = f(kwargv)
            l = loss_daily(
                predicted_confirmed_times=predicted_confirmed_times, 
                targets_daily=targets, 
                time_horizon=time_horizon,
                power=2.0)

            ave_pred = convert_timings_to_cumulative_daily(
                predicted_confirmed_times, time_horizon).mean(axis=0)

            loginfo = f'{-l}  ' + str(kwargv) + '\n'
            with open(f'logger_{c}.txt', 'a') as logfile:
                logfile.write(loginfo)

            # bayes_opt maximizes
            return - l

        return loss_function

    elif loss == 'multimodal_loss_daily':

        # here `extra_params` are weights
        if extra_params:
            weights = extra_params['weights']
        else:
            weights = np.ones(len(targets))

        def loss_function(**kwargv):
            preds = f(kwargv)
            l = multimodal_loss_daily(
                preds=preds, weights=weights, targets=targets, 
                time_horizon=time_horizon, power=2.0)
        
            # bayes_opt maximizes
            return - l

        return loss_function
    else:
        raise ValueError('Unknown loss function')

