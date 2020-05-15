import multiprocessing

'''
Default settings for model calibration
'''
TO_HOURS = 24.0

settings_data = {
    'verbose' : True,
    'use_households' : True,
    'data_start_date': '2020-03-10',
}

settings_simulation = {
    'n_init_samples': 20,  # initial random evaluations
    'n_iterations': 500,  # iterations of BO
    'simulation_roll_outs': 40, # roll-outs done in parallel per parameter setting
    'cpu_count':  multiprocessing.cpu_count(), # cpus used for parallel computation
    'dynamic_tracing' : True,
}

# parameter bounds
settings_model_param_bounds = {
    'betas': {
        'education': [0.0, 2.0],
        'social': [0.0, 2.0],
        'bus_stop': [0.0, 2.0],
        'office': [0.0, 2.0],
        'supermarket': [0.0, 2.0],
    },
    'beta_household': [0.0, 2.0],
    'mu': [0.0, 1.0]
}

settings_measures_param_bounds = {
    'p_stay_home': [0.0, 1.0],
}

# set testing parameters
settings_testing_params = {
    'testing_t_window': None,  # [set automatically in code]
    'testing_frequency': 1 * TO_HOURS,
    'test_reporting_lag': 2 * TO_HOURS,
    'tests_per_batch': None,  # [set automatically in code]
    'test_fpr': 0.0,
    'test_fnr': 0.0,
    'test_smart_delta': 3 * TO_HOURS,
    'test_smart_duration': 7 * TO_HOURS, 
    'test_smart_action': 'isolate',
    'test_smart_num_contacts': 10,
    'test_targets': 'isym',
    'test_queue_policy': 'fifo',
    'smart_tracing': None,
}

# BO acquisition function optimization (Knowledge gradient)
# default settings from botorch
settings_acqf = {
    'acqf_opt_num_fantasies': 64,
    'acqf_opt_num_restarts': 10,
    'acqf_opt_raw_samples': 256,
    'acqf_opt_batch_limit': 5,
    'acqf_opt_maxiter': 20,
}


# area codes
command_line_area_codes = {
    'GER' : {
        'TU': 'LK Tübingen',
        'KL': 'SK Kaiserslautern',
        'RH': 'LK Rheingau-Taunus-Kreis',
        'HB': 'LK Heinsberg',
        'TR': 'LK Tirschenreuth'
    },
    'CH' : {
        'SZ': 'SZ',     # Canton Schwyz
        'TI': 'TI',     # Canton Ticino
        'LU': 'LU',     # Canton Lucerne
        'VD': 'VD',     # Canton Vaud
    }
}				

# optimized model parameters
settings_optimized_town_params = {
    'GER': {
        'TU': { # dummy settings
            'betas': {
                'education': 0.2893,
                'social': 1.5401,
                'bus_stop': 0.9246,
                'office': 1.3179,
                'supermarket': 0.7882,
            },
            'beta_household': 0.0318,
        },
        'KL': {  # dummy settings
            'betas': {
                'education': 0.963,
                'social': 1.573,
                'bus_stop': 1.339,
                'office': 1.112,
                'supermarket': 1.671,
            },
            'beta_household': 1.442,
        },
        'RH': {  # dummy settings
            'betas': {
                'education': 0.245,
                'social': 0.049,
                'bus_stop': 1.932,
                'office': 0.719,
                'supermarket': 1.745,
            },
            'beta_household': 0.353,
        },
        'TR': {  # dummy settings
            'betas': {
                'education': 0.2359,
                'social': 1.8845,
                'bus_stop': 0.2532,
                'office': 0.9434,
                'supermarket': 0.6521,
            },
            'beta_household': 0.4105,
        },
    },
    'CH': {
        'SZ': {  # dummy settings
            'betas': {
                'education': 0.188,
                'social': 0.938,
                'bus_stop': 1.688,
                'office': 0.813,
                'supermarket': 0.563,
            },
            'beta_household': 0.688,
        },
        'TI': {  # dummy settings
            'betas': {
                'education': 0.320,
                'social': 2.303,
                'bus_stop': 1.564,
                'office': 2.791,
                'supermarket': 0.862,
            },
            'beta_household': 2.467,
        },
        'LU': {  # dummy settings
            'betas': {
                'education': 0.109,
                'social': 0.996,
                'bus_stop': 1.404,
                'office': 1.148,
                'supermarket': 0.781,
            },
            'beta_household': 0.756,
        },
        'VD': {  # dummy settings
            'betas': {
                'education': 0.0799,
                'social': 1.9321,
                'bus_stop': 1.6837,
                'office': 1.8394,
                'supermarket': 0.2438,
            },
            'beta_household': 1.9072,
        },
    }
}

# lockdown dates
settings_lockdown_dates = {
    'GER': {
        'start' : '03-23-2020',
        'end': '05-03-2020',
    },
    'CH': {
        'start': '03-16-2020',
        'end': '05-10-2020',
    },
}

