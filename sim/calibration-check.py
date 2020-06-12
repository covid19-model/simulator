import numpy as np
import pickle, math
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from lib.mobilitysim import MobilitySimulator
from lib.parallel import launch_parallel_simulations
from lib.distributions import CovidDistributions
from lib.data import collect_data_from_df
from lib.measures import (
    MeasureList, Interval,
    BetaMultiplierMeasureByType,
    SocialDistancingForAllMeasure, 
    SocialDistancingForPositiveMeasure,
    SocialDistancingForPositiveMeasureHousehold)
from lib.inference import gen_initial_seeds, downsample_cases
from lib.plot import Plotter
from lib.experiment import run_experiment, save_summary, load_summary, get_calibrated_params
from lib.calibration_settings import (
    command_line_area_codes, 
    calibration_lockdown_dates,
    calibration_model_param_bounds,
    calibration_mob_paths,
    calibration_start_dates,
    calibration_states,
    calibration_lockdown_beta_multipliers,
)

if __name__ == '__main__':

    full_scale = True
    dry_run = False
    plot = False
    random_repeats = 48 

    '''
    Don't change anything below
    '''

    c = 0
    np.random.seed(c)
    num_workers = multiprocessing.cpu_count()
    TO_HOURS = 24.0

    def standard_testing(max_time, cases):
        daily_increase = cases.sum(axis=1)[1:] - cases.sum(axis=1)[:-1]
        standard_testing_params = {
            'testing_t_window'    : [0.0, max_time], # in hours
            'testing_frequency'   : 1 * TO_HOURS,     # in hours
            'test_reporting_lag'  : 2 * TO_HOURS,     # in hours (actual and self-report delay)
            'tests_per_batch'     : int(daily_increase.max()), # test capacity based on empirical positive tests
            'test_fpr'            : 0.0, # test false positive rate
            'test_fnr'            : 0.0, # test false negative rate
            'test_smart_delta'    : 3 * TO_HOURS, # in hours
            'test_smart_duration' : 7 * TO_HOURS, # in hours
            'test_smart_action'   : 'isolate', 
            'test_smart_num_contacts'   : 10, 
            'test_targets'        : 'isym',
            'test_queue_policy'   : 'fifo',
            'smart_tracing'       : None, 
        }
        return standard_testing_params

    def params_to_strg(d):
        l = [
            f"{d['betas']['education']:8.4f}",
            f"{d['betas']['social']:8.4f}",
            f"{d['betas']['bus_stop']:8.4f}",
            f"{d['betas']['office']:8.4f}",
            f"{d['betas']['supermarket']:8.4f}",
            f"{d['beta_household']:8.4f}",
            f"{d['p_stay_home']:8.4f}",
        ]
        return ','.join(l)

    headerstr = ' educat | social | bus_st | office | superm | househ |  p_home'

    # Simulate for each town

    for country in ['GER', 'CH']:
        for area in calibration_mob_paths[country].keys():
            try:

                # start simulation when calibration started       
                mob_settings = calibration_mob_paths[country][area][1 if full_scale else 0] 
                start_date_calibration = calibration_start_dates[country][area]
                end_date_calibration = calibration_lockdown_dates[country]['end']

                # lockdown dates
                start_date_lockdown = calibration_lockdown_dates[country]['start']
                end_date_lockdown = calibration_lockdown_dates[country]['end']      

                # set time frame
                sim_days = (pd.to_datetime(end_date_calibration) - pd.to_datetime(start_date_calibration)).days
                max_time = TO_HOURS * sim_days # in hours

                # load mobility file
                with open(mob_settings, 'rb') as fp:
                    obj = pickle.load(fp)
                mob = MobilitySimulator(**obj)

                # case data + scaling
                unscaled_area_cases = collect_data_from_df(country=country, area=area, datatype='new',
                    start_date_string=start_date_calibration, end_date_string=end_date_calibration)

                sim_cases, unscaled_sim_cases = downsample_cases(unscaled_area_cases, mob)

                # distributions
                distributions = CovidDistributions(country=country)

                # seeds
                initial_seeds = gen_initial_seeds(
                    unscaled_sim_cases, 
                    downsampling=mob.downsample,
                    day=0)

                # calibrated parameters
                calibrated_params = get_calibrated_params(country, area)

                print(country, area, f'{mob.downsample}x', ' Days: ', sim_days,'  Start: ', start_date, '  End: ', end_date, )
                print('Start cases: Scaled : ', sim_cases[0].sum(), sim_cases[0], 'Unscaled : ' , unscaled_sim_cases[0].sum(), unscaled_sim_cases[0])
                print('End cases:   Scaled : ',  sim_cases[-1].sum(), sim_cases[-1], 'Unscaled : ' , unscaled_sim_cases[-1].sum(), unscaled_sim_cases[-1])
                print('Scaled seeds:  ', initial_seeds)
                print('Unscaled seeds:',  gen_initial_seeds(
                    unscaled_sim_cases, 
                    downsampling=1,
                    day=0))

                print(headerstr)
                print(params_to_strg(calibrated_params))

                # measures
                days_until_lockdown = (pd.to_datetime(start_date_lockdown) - pd.to_datetime(start_date)).days

                measure_list =  MeasureList([
                    SocialDistancingForPositiveMeasure(
                        t_window=Interval(0.0, max_time), p_stay_home=1.0),
                    SocialDistancingForPositiveMeasureHousehold(
                        t_window=Interval(0.0, max_time), p_isolate=1.0),
                    BetaMultiplierMeasureByType(
                            t_window=Interval(days_until_lockdown * TO_HOURS, max_time * TO_HOURS), 
                            beta_multiplier=calibration_lockdown_beta_multipliers),
                    SocialDistancingForAllMeasure(
                        t_window=Interval(days_until_lockdown * TO_HOURS, max_time * TO_HOURS), 
                        p_stay_home=calibrated_params['p_stay_home'])
                ])

                # testing
                testing_params = standard_testing(max_time, sim_cases)
                print('Test capacity: Scaled: ', testing_params['tests_per_batch'], 
                    ' Unscaled: ', standard_testing(max_time, unscaled_sim_cases)['tests_per_batch'])
                print()

                if not dry_run:
                    # run simulations
                    model_params = {
                        'betas' : calibrated_params['betas'],
                        'beta_household' : calibrated_params['beta_household'],
                    }

                    summary = launch_parallel_simulations(
                        mob_settings=mob_settings, 
                        distributions=distributions, 
                        random_repeats=random_repeats, 
                        cpu_count=num_workers, 
                        params=model_params, 
                        initial_seeds=initial_seeds, 
                        testing_params=testing_params, 
                        measure_list=measure_list, 
                        max_time=max_time, 
                        num_people=mob.num_people, 
                        num_sites=mob.num_sites, 
                        site_loc=mob.site_loc, 
                        home_loc=mob.home_loc,
                        lazy_contacts=True,
                        verbose=False)

                    appdx = 'full' if full_scale else 'downscaled'
                    save_summary(summary, 'summary-calib--{}-{}--{}.pk'.format(country, area, appdx))
            except FileNotFoundError:
                print(country, area, ' file not found.')


    # Plot


    ymax = {
        'GER' : {
            'TU' : 2000,
            'KL' : 800,
            'RH' : 1000,
            'TR' : 2000,
        },
        'CH' : {
            'VD' : 2000,
            'BE' : 600,
            'TI' : 500,
            'JU' : 500,
        }
    }

    if plot:
        for country in ['GER', 'CH']:
            for area in calibration_mob_paths[country].keys():
                
                try:
                    print(country, area)

                    # start simulation when calibration started       
                    mob_settings = calibration_mob_paths[country][area][1 if full_scale else 0] 
                    start_date_calibration = calibration_start_dates[country][area]
                    end_date_calibration = calibration_lockdown_dates[country]['end']

                    # lockdown dates
                    start_date_lockdown = calibration_lockdown_dates[country]['start']
                    end_date_lockdown = calibration_lockdown_dates[country]['end']      

                    # set time frame
                    sim_days = (pd.to_datetime(end_date_calibration) - pd.to_datetime(start_date_calibration)).days
                    max_time = TO_HOURS * sim_days # in hours

                    # load mobility file
                    with open(mob_settings, 'rb') as fp:
                        obj = pickle.load(fp)
                    mob = MobilitySimulator(**obj)

                    # case data + scaling
                    unscaled_area_cases = collect_data_from_df(country=country, area=area, datatype='new',
                        start_date_string=start_date_calibration, end_date_string=end_date_calibration)

                    sim_cases, unscaled_sim_cases = downsample_cases(unscaled_area_cases, mob)

                    appdx = 'full' if full_scale else 'downscaled'
                    loadstr = 'summary-calib--{}-{}--{}.pk'.format(country, area, appdx)
                    summary = load_summary(loadstr)
                    print(loadstr)

                    
                    # ymax depending on full scale vs downsampling
                    ym = ymax[country][area] / (mob.downsample) 

                    plotter = Plotter()
                    plotter.plot_positives_vs_target(
                        summary, sim_cases.sum(axis=1), 
                        title='Calibration period', 
                        filename='calibration-{}-{}-{}-{}'.format('only-calib' if calibration_period_only else 'full',country, area, appdx),
                        figsize=(6, 4),
                        start_date=start_date,
                        errorevery=1, acc=1000, 
                        ymax=int(ym),
                    )

                    plotter.plot_age_group_positives_vs_target(
                        summary, sim_cases, 
                        ytitle=f'{country}-{area}',
                        filename='calibration-{}-{}-{}-age-{}'.format('only-calib' if calibration_period_only else 'full', country, area, appdx),
                        figsize=(16, 2.5),
                        start_date=start_date,
                        errorevery=1, acc=1000, 
                        ymax=int(ym / 4),
                    )

                except FileNotFoundError:
                    print(country, area, ' file not found.')
    else:
        print('Plotting mode off.')
            
