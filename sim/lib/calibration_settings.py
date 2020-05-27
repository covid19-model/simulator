
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
    'lazy_contacts' : True,
}

# parameter bounds
beta_upper_bound = 3.5
settings_model_param_bounds = {
    'betas': {
        'education': [0.0, beta_upper_bound],
        'social': [0.0, beta_upper_bound],
        'bus_stop': [0.0, beta_upper_bound],
        'office': [0.0, beta_upper_bound],
        'supermarket': [0.0, beta_upper_bound],
    },
    'beta_household': [0.0, beta_upper_bound],
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
        'BE': 'BE',     # Canton Bern
        'VD': 'VD',     # Canton Vaud
        'JU': 'JU',     # Canton Jura
    }
}				

# lockdown dates
settings_lockdown_dates = {
    'GER': {
        'start' : '2020-03-23',
        'end': '2020-05-03',
    },
    'CH': {
        'start': '2020-03-16',
        'end': '2020-05-10',
    },
}

# settings path; calibration start date; calibration end date;
calibration_setting_paths = {
    'GER': {
        'TU': ['lib/mobility/Tubingen_settings_10.pk', '2020-03-12', '2020-03-28'],
        'KL': ['lib/mobility/Kaiserslautern_settings_5.pk', '2020-03-15', '2020-03-28'],
        'RH': ['lib/mobility/Ruedesheim_settings_10.pk', '2020-03-10', '2020-03-28'],
        'TR': ['lib/mobility/Tirschenreuth_settings_10.pk', '2020-03-13', '2020-03-28'],
    },
    'CH': {
        'VD': ['lib/mobility/Lausanne_settings_10.pk', '2020-03-07', '2020-03-21'],
        'BE': ['lib/mobility/Bern_settings_5.pk', '2020-03-06', '2020-03-21'],
        'TI': ['lib/mobility/Locarno_settings_2.pk', '2020-03-09', '2020-03-21'],
        'JU': ['lib/mobility/Jura_settings_10.pk', '2020-03-09', '2020-03-21'],
    }
}
# copy of above but with full scale town versions
calibration_setting_paths = {
    'GER': {
        'TU': ['lib/mobility/Tubingen_settings_1.pk', '2020-03-12', '2020-03-28'],
        'KL': ['lib/mobility/Kaiserslautern_settings_1.pk', '2020-03-15', '2020-03-28'],
        'RH': ['lib/mobility/Ruedesheim_settings_1.pk', '2020-03-10', '2020-03-28'],
        'TR': ['lib/mobility/Tirschenreuth_settings_1.pk', '2020-03-13', '2020-03-28'],
    },
    'CH': {
        'VD': ['lib/mobility/Lausanne_settings_1.pk', '2020-03-07', '2020-03-21'],
        'BE': ['lib/mobility/Bern_settings_1.pk', '2020-03-06', '2020-03-21'],
        'TI': ['lib/mobility/Locarno_settings_1.pk', '2020-03-09', '2020-03-21'],
        'JU': ['lib/mobility/Jura_settings_1.pk', '2020-03-09', '2020-03-21'],
    }
}

# calibration states loaded for calibrated parameters
calibration_states = {
    'GER': {
        'TU': 'logs/calibration_tu0_state.pk',
        'KL': 'logs/calibration_kl0_state.pk',
        'RH': 'logs/calibration_rh0_state.pk',
        'TR': 'logs/calibration_tr0_state.pk',
    },
    'CH': {
        'VD': 'logs/calibration_vd0_state.pk',
        'BE': 'logs/calibration_be0_state.pk',
        'TI': 'logs/calibration_ti0_state.pk',
        'JU': 'logs/calibration_ju0_state.pk',
    }
}
