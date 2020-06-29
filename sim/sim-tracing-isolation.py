
import sys, os
if '..' not in sys.path:
    sys.path.append('..')

import numpy as np
import random as rd
import pandas as pd
import pickle
import multiprocessing
from lib.measures import *
from lib.experiment import Experiment, options_to_str
from lib.calibrationSettings import calibration_lockdown_dates, calibration_mob_paths, calibration_states
from lib.calibrationFunctions import get_calibrated_params

TO_HOURS = 24.0

if __name__ == '__main__':

    name = 'tracing-isolation'
    end_date = '2020-07-31'
    random_repeats = 6
    full_scale = False
    seed_summary_path = None
    dry_run = False
    verbose = True

    # seed
    c = 0
    np.random.seed(0)
    rd.seed(0)

    # experiment parameters
    isolated_days = [7, 14] # how many days selected people have to stay in isolation
    contacts_isolated = [10, 25] # how many contacts are isolated in the `test_smart_delta` window
    policies = ['basic', 'advanced'] # contact tracing policies
    
    # configure the experiment for each country
    for country, areas in calibration_mob_paths.items():
        for area, mob_settings in areas.items():
            
            # check calibration state
            if not os.path.isfile(calibration_states[country][area]):
                print(f'{country}-{area} calibration not found.')
                continue

            # start simulation when lockdown ends
            start_date = calibration_lockdown_dates[country]['end']

            # load social distancing parameter
            p_stay_home = get_calibrated_params(country=country, area=area,
                                                multi_beta_calibration=False)['p_stay_home']

            # create experiment object
            exp_name = f'{name}-{country}-{area}'
    
            experiment = Experiment(
                name=exp_name,
                start_date=start_date,
                end_date=end_date,
                random_repeats=random_repeats,
                full_scale=full_scale,
                verbose=verbose,
            )
            
            # baseline
            experiment.add(
                description='baseline',
                country=country,
                area=area,
                measure_list=[],
                seed_summary_path=seed_summary_path,
                full_scale=full_scale)
        
            # contact tracing experiment for various options
            for isolate_days in isolated_days:
                for contacts in contacts_isolated:
                    for policy in policies:

                        # measures
                        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                        
                        m = [SocialDistancingForSmartTracing(
                                t_window=Interval(0.0, TO_HOURS * max_days), 
                                p_stay_home=1.0, 
                            test_smart_duration=TO_HOURS * isolate_days)
                        ]

                        # update testing params
                        def test_update(d):
                            d['test_smart_delta'] =  3 * TO_HOURS # 3 day time window considered for inspecting contacts
                            d['test_smart_action'] = 'isolate' # isolate traced individuals
                            d['test_targets'] = 'isym' 
                            d['smart_tracing'] = policy
                            d['test_smart_num_contacts'] = contacts
                            return d

                        sim_descr = options_to_str(
                            isolate_days=isolate_days, 
                            contacts=contacts, 
                            policy=policy)
                            
                        experiment.add(
                            description=sim_descr,
                            country=country,
                            area=area,
                            measure_list=m,
                            test_update=test_update,
                            seed_summary_path=seed_summary_path,
                            full_scale=full_scale)
            
            # execute all simulations
            print(f'{exp_name} configuration done.')

            if not dry_run:
                experiment.run_all()

