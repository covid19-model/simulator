import sys
if '..' not in sys.path:
    sys.path.append('..')

import argparse
from lib.calibration_settings import *


def make_calibration_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="set seed")
    parser.add_argument("--filename", help="set filename; default is `calibartion_{seed}` ")
    parser.add_argument("--not_verbose", action="store_true", help="not verbose; default is verbose")

    # BO
    parser.add_argument("--ninit", type=int, default=settings_simulation['n_init_samples'],
        help="update default number of quasi-random initial evaluations")
    parser.add_argument("--niters", type=int, default=settings_simulation['n_iterations'],
        help="update default number of BO iterations")
    parser.add_argument("--rollouts", type=int, default=settings_simulation['simulation_roll_outs'],
        help="update default number of parallel simulation rollouts")
    parser.add_argument("--cpu_count", type=int, default=settings_simulation['cpu_count'],
        help="update default number of cpus used for parallel simulation rollouts")
    parser.add_argument("--load", 
        help="specify path to a BO state to be loaded as initial observations, e.g. 'logs/calibration_0_state.pk'")

    # SD tuning
    parser.add_argument("--measures_optimized", action="store_true",
        help="when passed, BO optimizes `p_stay_home` for SocialDistancingForAllMeasure at given starting date")
    parser.add_argument("--measures_close", nargs="+", 
        help="when `--measures_optimized` is active, closes all site types passed after this argument")
 

    # data
    parser.add_argument("--mob", 
        help="specify path to mobility settings for trace generation, e.g. 'lib/tu_settings_10.pk'")
    parser.add_argument("--country", 
        help="specify country indicator for data import")
    parser.add_argument("--area", 
        help="specify area indicator for data import")
    parser.add_argument("--start",
        help="set starting date for which case data is retrieved "
             "e.g. '2020-03-10'")
    parser.add_argument("--end",
        help="set end date for which case data is retrieved "
             "e.g. '2020-03-26'")

    # simulation
    parser.add_argument("--no_households", action="store_true",
                        help="no households should be used for simulation")
    parser.add_argument("--no_dynamic_tracing", action="store_true",
                        help="no dynamic online computation of mobility traces (default is online)")
    parser.add_argument("--endsimat", type=int,
                        help="for debugging: specify number of days after which simulation should be cut off")
    parser.add_argument("--testingcap", type=int,
                        help="overwrite default unscaled testing capacity as provided by MobilitySimulator")


    # acquisition function optimization
    parser.add_argument("--acqf_opt_num_fantasies", type=int, default=settings_acqf['acqf_opt_num_fantasies'],
        help="update default for acquisition function optim.: number of fantasies")
    parser.add_argument("--acqf_opt_num_restarts", type=int, default=settings_acqf['acqf_opt_num_restarts'],
        help="update default for acquisition function optim.: number of restarts")
    parser.add_argument("--acqf_opt_raw_samples", type=int, default=settings_acqf['acqf_opt_raw_samples'],
        help="update default for acquisition function optim.: number of raw samples")
    parser.add_argument("--acqf_opt_batch_limit", type=int, default=settings_acqf['acqf_opt_batch_limit'],
        help="update default for acquisition function optim.: batch limit")
    parser.add_argument("--acqf_opt_maxiter", type=int, default=settings_acqf['acqf_opt_maxiter'],
        help="update default for acquisition function optim.: maximum iteraitions")

    return parser
