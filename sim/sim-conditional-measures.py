
import sys
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
from lib.measures import *
from lib.experiment import Experiment, options_to_str, process_command_line
from lib.calibrationSettings import calibration_lockdown_beta_multipliers
from lib.calibrationFunctions import get_calibrated_params

TO_HOURS = 24.0

if __name__ == '__main__':

    # command line parsing
    args = process_command_line()
    country = args.country
    area = args.area
    cpu_count = args.cpu_count
    continued_run = args.continued

    name = 'conditional-measures'
    start_date = '2021-01-01'
    end_date = '2021-05-01'
    random_repeats = 100
    full_scale = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = {}
    expected_daily_base_expo_per100k = 5 / 7
    condensed_summary = True

    # seed
    c = 0
    np.random.seed(c)
    rd.seed(c)

    calibrated_params = get_calibrated_params(country=country, area=area)

    # experiment parameters
    # Measures become active only if positive tests per week per 100k people exceed `max_pos_tests_per_week_per_100k`.
    # `intervention_times` can be set in order to allow changes in the policy only at discrete times,
    # if `None` measures can become active at any point.
    # `p_stay_home` determines how likely people are to follow the social distancing measures if they are active.
    # `beta_multiplier` determines scaling for `beta` when measure is active

    max_pos_tests_per_week_per_100k = [50]
    if args.p_adoption is not None:
        p_compliances = [args.p_adoption]
    else:
        p_compliances = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]

    if args.smoke_test:
        start_date = '2021-01-01'
        end_date = '2021-02-15'
        random_repeats = 1
        full_scale = False
        p_compliances = [0.75]

    # create experiment object
    experiment_info = f'{name}-{country}-{area}'
    experiment = Experiment(
        experiment_info=experiment_info,
        start_date=start_date,
        end_date=end_date,
        random_repeats=random_repeats,
        cpu_count=cpu_count,
        full_scale=full_scale,
        verbose=verbose,
    )

    for max_incidence in max_pos_tests_per_week_per_100k:
        for p_compliance in p_compliances:
            # conditional measures experiment
            max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

            m = [
                UpperBoundCasesSocialDistancing(t_window=Interval(0.0, TO_HOURS * max_days),
                                                p_stay_home=p_compliance,
                                                max_pos_tests_per_week_per_100k=max_incidence,
                                                intervention_times=None,
                                                init_active=False),
                ]

            simulation_info = options_to_str(
                max_incidence=max_incidence,
                p_compliance=p_compliance,
            )

            experiment.add(
                simulation_info=simulation_info,
                country=country,
                area=area,
                measure_list=m,
                seed_summary_path=seed_summary_path,
                set_calibrated_params_to=calibrated_params,
                set_initial_seeds_to=set_initial_seeds_to,
                full_scale=full_scale,
                expected_daily_base_expo_per100k=expected_daily_base_expo_per100k)

    print(f'{experiment_info} configuration done.')

    # execute all simulations
    experiment.run_all()

