import sys
if '..' not in sys.path:
    sys.path.append('..')

import argparse
from lib.calibrationSettings import *


def make_calibration_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="set seed")
    parser.add_argument("--filename", help="set filename; default is `calibartion_{seed}` ")
    parser.add_argument("--not_verbose", action="store_true", help="not verbose; default is verbose")
    parser.add_argument("--smoke_test", action="store_true", help="debug mode; stops quickly")
    parser.add_argument("--plot-fit", action="store_true", help="plots model fit at each improvement and stores inside `logs/`")

    # BO
    parser.add_argument("--ninit", type=int, default=calibration_simulation['n_init_samples'],
        help="update default number of quasi-random initial evaluations")
    parser.add_argument("--niters", type=int, default=calibration_simulation['n_iterations'],
        help="update default number of BO iterations")
    parser.add_argument("--rollouts", type=int, default=calibration_simulation['simulation_roll_outs'],
        help="update default number of parallel simulation rollouts")
    parser.add_argument("--cpu_count", type=int, default=calibration_simulation['cpu_count'],
        help="update default number of cpus used for parallel simulation rollouts")
    parser.add_argument("--load", 
        help="specify path to a BO state to be loaded as initial observations, e.g. 'logs/calibration_0_state.pk'")

    parser.add_argument("--model-multi-output-simulator", action="store_true",
                        help="flag to model every day of simulator output individually using an GP")
    parser.add_argument("--model-noise-via-sem", action="store_true",
                        help="model simulator noise using standard error of mean")
    parser.add_argument("--normalize-cases", action="store_true",
                        help="use unnormalized case numbers; if not passed, divides all case numbers by target on t=T")
    parser.add_argument("--estimate_mobility_reduction", action="store_true",
                        help="whether to estimate p_stay_home; if not set, uses Google mobility report")

    parser.add_argument("--multi-beta-calibration", action="store_true",
                        help="flag to calibrate an individual beta parameter for each site category/type")
    parser.add_argument("--per-age-group-objective", action="store_true",
                        help="flag to calibrate based on per age-group objective")
    parser.add_argument("--use-fixed-lockdown-beta-multipliers", action="store_true",
                        help="uses beta multipliers at sites during the lockdown period")
    parser.add_argument("--log-objective", action="store_true",
                        help="log-MSE instead of MSE")
    
    # data
    parser.add_argument("--mob", 
        help="update path to mobility settings for trace generation")
    parser.add_argument("--country", required=True,
        help="specify country indicator for data import")
    parser.add_argument("--area",  required=True,
        help="specify area indicator for data import")
    parser.add_argument("--start",
        help="update starting date for which case data is retrieved "
             "e.g. '2020-03-10'")
    parser.add_argument("--end",
        help="update end date for which case data is retrieved "
             "e.g. '2020-03-26'")

    # simulation
    parser.add_argument("--no_households", action="store_true",
                        help="no households should be used for simulation")
    parser.add_argument("--testingcap", type=int,
                        help="overwrite default unscaled testing capacity as provided by MobilitySimulator")

    # acquisition function optimization
    parser.add_argument("--acqf_opt_num_fantasies", type=int, default=calibration_acqf['acqf_opt_num_fantasies'],
        help="update default for acquisition function optim.: number of fantasies")
    parser.add_argument("--acqf_opt_num_restarts", type=int, default=calibration_acqf['acqf_opt_num_restarts'],
        help="update default for acquisition function optim.: number of restarts")
    parser.add_argument("--acqf_opt_raw_samples", type=int, default=calibration_acqf['acqf_opt_raw_samples'],
        help="update default for acquisition function optim.: number of raw samples")
    parser.add_argument("--acqf_opt_batch_limit", type=int, default=calibration_acqf['acqf_opt_batch_limit'],
        help="update default for acquisition function optim.: batch limit")
    parser.add_argument("--acqf_opt_maxiter", type=int, default=calibration_acqf['acqf_opt_maxiter'],
        help="update default for acquisition function optim.: maximum iteraitions")

    return parser
