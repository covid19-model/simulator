
import sys
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
from lib.measures import *
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.calibrationFunctions import get_calibrated_params
from lib.calibrationSettings import calibration_lockdown_beta_multipliers

TO_HOURS = 24.0

if __name__ == '__main__':
    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area
    cpu_count = args.cpu_count
    continued_run = args.continued

    name = 'vulnerable-groups'
    start_date = '2021-01-01'
    end_date = '2021-05-01'
    random_repeats = 100
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 5 / 7
    condensed_summary = True

    # experiment parameters
    if args.p_adoption is not None:
        p_compliances = [args.p_adoption]
    else:
        p_compliances = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]

    if args.smoke_test:
        start_date = '2021-01-01'
        end_date = '2021-02-15'
        random_repeats = 1
        p_compliances = [0.75]
        full_scale = False

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    # Load calibrated parameters up to `maxBOiters` iterations of BO
    # maxBOiters = 40 if area in ['BE', 'JU', 'RH'] else None
    calibrated_params = get_calibrated_params(country=country, area=area,
                                              multi_beta_calibration=False,
                                              maxiters=None,
                                              estimate_mobility_reduction=False)

    # create experiment object
    experiment_info = f'{name}-{country}-{area}'
    experiment = Experiment(
        experiment_info=experiment_info,
        start_date=start_date,
        end_date=end_date,
        random_repeats=random_repeats,
        cpu_count=cpu_count,
        full_scale=full_scale,
        condensed_summary=condensed_summary,
        continued_run=continued_run,
        verbose=verbose,
    )

    for p_compliance in p_compliances:
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

        m = [
            SocialDistancingByAgeMeasure(
                t_window=Interval(0.0, TO_HOURS * max_days),
                p_stay_home=(
                    [0.0, 0.0, 0.0, 0.0, p_compliance, p_compliance] if country == 'GER' else
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_compliance, p_compliance, p_compliance]
                )),

            APrioriBetaMultiplierMeasureByType(beta_multiplier=calibration_lockdown_beta_multipliers)
            ]

        simulation_info = options_to_str(p_compliance=p_compliance,
                                         beta_multiplier=calibration_lockdown_beta_multipliers['education'])

        experiment.add(
            simulation_info=simulation_info,
            country=country,
            area=area,
            measure_list=m,
            lockdown_measures_active=False,
            seed_summary_path=seed_summary_path,
            set_initial_seeds_to=set_initial_seeds_to,
            set_calibrated_params_to=calibrated_params,
            full_scale=full_scale,
            expected_daily_base_expo_per100k=expected_daily_base_expo_per100k)

    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

