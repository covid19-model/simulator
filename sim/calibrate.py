import sys
import argparse
if '..' not in sys.path:
    sys.path.append('..')

import pandas as pd
import numpy as np
import networkx as nx
import copy
import scipy as sp
import math
import seaborn
import pickle
import warnings
import matplotlib
import re
import multiprocessing
import torch

from botorch import fit_gpytorch_model
from botorch.exceptions import BadInitialCandidatesWarning
import botorch.utils.transforms as transforms
from lib.calibrationFunctions import make_bayes_opt_functions, pdict_to_parr, parr_to_pdict, CalibrationLogger, save_state, load_state, gen_initial_seeds
from lib.kg import qKnowledgeGradient
import time, pprint

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from lib.mobilitysim import MobilitySimulator
from lib.dynamics import DiseaseModel
from bayes_opt import BayesianOptimization
from lib.parallel import *
from lib.distributions import CovidDistributions
from lib.plot import Plotter
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler


from lib.mobilitysim import MobilitySimulator
from lib.calibrationParser import make_calibration_parser

if __name__ == '__main__':

    '''
    Command line arguments
    '''

    parser = make_calibration_parser()
    args = parser.parse_args()
    seed = args.seed or 0
    args.filename = args.filename or f'calibration_{seed}'

    if args.smoke_test:
        args.ninit = 2
        args.niters = 1
        args.rollouts = 2
        args.start = "2020-03-10"
        args.end = "2020-03-17"

        
    '''
    Genereate essential functions for Bayesian optimization
    '''
    
    (objective, 
     generate_initial_observations, 
     initialize_model, 
     optimize_acqf_and_get_observation, 
     case_diff,
     unnormalize_theta,
     header) = make_bayes_opt_functions(args=args)

    # logger
    logger = CalibrationLogger(
        filename=args.filename, 
        multi_beta_calibration=args.multi_beta_calibration,
        estimate_mobility_reduction=args.estimate_mobility_reduction,
        verbose=not args.not_verbose)
    logger.log_initial_lines(header)

    # if specified, load initial training data
    if args.load:

        # load initial observations 
        state = load_state(args.load)
        loaded_theta = state['train_theta']
        loaded_G = state['train_G']
        loaded_G_sem = state['train_G_sem']
        n_loaded = state['train_theta'].shape[0]

        # if any initialization remains to be done, evaluate remaining initial points
        train_theta, train_G, train_G_sem, best_observed_obj, best_observed_idx = generate_initial_observations(
            n=args.ninit, logger=logger, loaded_init_theta=loaded_theta, loaded_init_G=loaded_G, loaded_init_G_sem=loaded_G_sem)
        n_bo_iters_loaded = max(n_loaded - args.ninit, 0)

    # else, if not specified, generate initial training data
    else:
        train_theta, train_G, train_G_sem, best_observed_obj, best_observed_idx = generate_initial_observations(
            n=args.ninit, logger=logger, loaded_init_theta=None, loaded_init_G=None, loaded_init_G_sem=None)
        n_bo_iters_loaded = 0

    # init model based on initial observations
    mll, model = initialize_model(train_theta, train_G, train_G_sem)

    # run n_iterations rounds of Bayesian optimization after the initial random batch
    for tt in range(n_bo_iters_loaded, args.niters):

        t0 = time.time()

        # fit the GP model
        fit_gpytorch_model(mll)

        # define acquisition function based on fitted GP
        acqf = qKnowledgeGradient(
            model=model,
            objective=objective,
            num_fantasies=args.acqf_opt_num_fantasies,
            inner_sampler=SobolQMCNormalSampler(
                num_samples=512, resample=False, collapse_batch_dims=True 
                # default internally was num_samples=128, increased for higher
                # accuracy in objective evaluation
            )
        )
        
        # optimize acquisition and get new observation via simulation at selected parameters
        new_theta, new_G, new_G_sem = optimize_acqf_and_get_observation(
            acq_func=acqf,
            args=args)
            
        # concatenate observations
        train_theta = torch.cat([train_theta, new_theta.unsqueeze(0)], dim=0) 
        train_G = torch.cat([train_G, new_G.unsqueeze(0)], dim=0)
        train_G_sem = torch.cat([train_G_sem, new_G_sem.unsqueeze(0)], dim=0)
        
        # update progress
        train_G_objectives = objective(train_G)
        best_observed_idx = train_G_objectives.argmax()
        best_observed_obj = train_G_objectives[best_observed_idx].item()
        
        # re-initialize the models so they are ready for fitting on next iteration
        mll, model = initialize_model(
            train_theta, 
            train_G, 
            train_G_sem,
        )

        walltime = time.time() - t0
        
        # log
        logger.log(
            i=tt,
            time=walltime,
            best=best_observed_obj,
            case_diff=case_diff(new_G),
            objective=objective(new_G).item(),
            theta=unnormalize_theta(new_theta.detach().squeeze())
        )

        # save state
        state = {
            'train_theta' : train_theta,
            'train_G' : train_G,
            'train_G_sem'  : train_G_sem,
            'best_observed_obj': best_observed_obj,
            'best_observed_idx': best_observed_idx
        }
        save_state(state, args.filename)

    # print best parameters
    print()
    print('FINISHED.')
    print('Best objective:  ', best_observed_obj)
    print('Best parameters:')
    
    # scale back to simulation parameters (from unit cube parameters in BO)
    normalized_calibrated_params = train_theta[best_observed_idx]
    calibrated_params = unnormalize_theta(normalized_calibrated_params)
    pprint.pprint(parr_to_pdict(parr=calibrated_params, 
        multi_beta_calibration=args.multi_beta_calibration, 
        estimate_mobility_reduction=args.estimate_mobility_reduction))


