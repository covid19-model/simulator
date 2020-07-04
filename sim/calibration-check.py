import sys, os
if '..' not in sys.path:
    sys.path.append('..')

import random as rd
import pandas as pd
import pickle
from lib.measures import *
from lib.data import collect_data_from_df
from lib.experiment import Experiment, options_to_str, process_command_line, load_summary
from lib.calibrationSettings import calibration_lockdown_dates, calibration_start_dates, calibration_mob_paths
from lib.calibrationFunctions import get_calibrated_params, downsample_cases
from beta_scaling_factors import *


if __name__ == '__main__':

    name = 'calibration-check'
    random_repeats = 96
    plot = True
    verbose = True
    seed_summary_path = None
    set_initial_seeds_to = None
    debugmode = False

    # command line parsing
    parser = process_command_line(return_parser=True)
    parser.add_argument("--plot_only", default=False)
    parser.add_argument("--no_agegroups", default=False)
    args = parser.parse_args()
    country = args.country
    area = args.area
    plot_only = args.plot_only
    no_agegroups = args.no_agegroups

    # Simulations
    exps = {'downscaled': {'full_scale': False, 'beta_scaling': 1.0},
            'full': {'full_scale': True, 'beta_scaling': 1.0},
            'full-ave_total_contact_time-scaled-beta': {'full_scale': True,
                                                        'beta_scaling': ave_total_contact_time[country][area]},
            'full-ave_ave_contact_time-scaled-beta': {'full_scale': True,
                                                      'beta_scaling': ave_ave_contact_time[country][area]},
            'full-ave_ave_contact_time_unique-scaled-beta': {'full_scale': True,
                                                             'beta_scaling': ave_ave_contact_time_unique[country][area]}

            }

    '''
    Don't change anything below
    '''

    # seed
    c = 0
    np.random.seed(0)
    rd.seed(0)

    start_date = calibration_start_dates[country][area]
    end_date = calibration_lockdown_dates[country]['end']

    if debugmode:
        random_repeats = 2
        end_date = pd.to_datetime(calibration_start_dates[country][area]) + pd.to_timedelta(1, unit='D')
        end_date = end_date.strftime('%Y-%m-%d')

    # create experiment object
    experiment_info = f'{name}-{country}-{area}'
    experiment = Experiment(
        experiment_info=experiment_info,
        start_date=start_date,
        end_date=end_date,
        random_repeats=random_repeats,
        full_scale=None,
        verbose=verbose,
    )

    summary_paths = []
    for exp, expparams in exps.items():
        calibrated_params = get_calibrated_params(country=country, area=area, multi_beta_calibration=False)
        calibrated_params['beta_site'] = expparams['beta_scaling'] * calibrated_params['beta_site']

        simulation_info = options_to_str(exp=exp, beta_scaling=expparams['beta_scaling'])
        # FIXME: If run again, use the following, don't change now!
        # scaling = expparams['beta_scaling']
        #  simulation_info = options_to_str(exp=exp+f'={scaling}')

        summary_path = experiment_info + '/' + experiment_info + '-' + simulation_info
        summary_paths.append(summary_path)

        if not os.path.exists('summaries/' + summary_path + '.pk'):
            experiment.add(
                simulation_info=simulation_info,
                country=country,
                area=area,
                test_update=None,
                measure_list=[],  # set automatically during lockdown
                seed_summary_path=seed_summary_path,
                set_initial_seeds_to=set_initial_seeds_to,
                set_calibrated_params_to=calibrated_params,
                full_scale=expparams['full_scale'])
            print(f'{experiment_info} configuration done.')
        else:
            print(f'Summary file exists already, skipping experiment {experiment_info}-{simulation_info}')

    if not plot_only:
        experiment.run_all()
    else:
        print('Simulations were not run. Trying to produce plots from existing summaries.')

    if plot:
        from lib.plot import Plotter
        ymax = {
            'GER': {
                'TU': 1200,
                'KL': 800,
                'RH': 1000,
                'TR': 2000,
            },
            'CH': {
                'VD': 2000,
                'BE': 600,
                'TI': 500,
                'JU': 500,
            }
        }

        for summary_path, exp in zip(summary_paths, exps.values()):
            try:
                _, filename = os.path.split(summary_path)
                print('Plotting: ' + filename)
                resulttuple = load_summary(summary_path+'.pk')
                summary = resulttuple[1]

                mob_settings_paths = calibration_mob_paths[country][area][1 if exp['full_scale'] else 0]
                with open(mob_settings_paths, 'rb') as fp:
                    mob_settings = pickle.load(fp)

                area_cases = collect_data_from_df(country=country,
                                                 area=area,
                                                 datatype='new',
                                                 start_date_string=start_date,
                                                 end_date_string=end_date)

                sim_cases = downsample_cases(area_cases, mob_settings)      # only downscaling due LK data for cities

                plotter = Plotter()
                plotter.plot_positives_vs_target(
                    summary, sim_cases.sum(axis=1),
                    title='Calibration period',
                    filename=filename,
                    figsize=(6, 4),
                    start_date=start_date,
                    errorevery=1, acc=1000,
                    ymax=int(ymax[country][area])
                )

                if not no_agegroups:
                    plotter.plot_age_group_positives_vs_target(
                        summary, sim_cases,
                        ytitle=f'{country}-{area}',
                        filename=filename + '-age',
                        figsize=(16, 2.5),
                        start_date=start_date,
                        errorevery=1, acc=1000,
                        ymax=int(ymax[country][area] / 4))
            except FileNotFoundError:
                print(f'{experiment_info}-{simulation_info}. File not found.')
    else:
        print('Plotting mode off.')
