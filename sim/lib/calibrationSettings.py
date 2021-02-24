
import multiprocessing
from lib.mobility_reduction import get_mobility_reduction



'''
Default settings for model calibration
'''

TO_HOURS = 24.0

calibration_data = {
    'verbose' : True,
    'use_households' : True,
    'data_start_date': '2020-03-10',
}

calibration_simulation = {
    'n_init_samples': 10,  # initial random evaluations
    'n_iterations': 90,  # iterations of BO
    'simulation_roll_outs': 96, # roll-outs done in parallel per parameter setting
    'cpu_count':  multiprocessing.cpu_count(), # cpus used for parallel computation
}

# parameter bounds

beta_upper_bound_site = 0.08
beta_upper_bound_household = 0.08

calibration_model_param_bounds_single = {
    'beta_site': [0.0, beta_upper_bound_site],
    'beta_household': [0.0, beta_upper_bound_household],
}

calibration_model_param_bounds_multi = {
    'betas': {
        'education': [0.0, beta_upper_bound_site],
        'social': [0.0, beta_upper_bound_site],
        'bus_stop': [0.0, beta_upper_bound_site],
        'office': [0.0, beta_upper_bound_site],
        'supermarket': [0.0, beta_upper_bound_site],
    },
    'beta_household': [0.0, beta_upper_bound_household],
}


# set testing parameters
calibration_testing_params = {
    'testing_t_window': None,  # [set automatically in code]
    'testing_frequency': 1 * TO_HOURS,
    'test_reporting_lag': 2 * TO_HOURS,
    'tests_per_batch': 100000, 
    'test_fpr': 0.0,
    'test_fnr': 0.0,
    'test_targets': 'isym',
    'test_queue_policy': 'fifo',

    # smart tracing
    'smart_tracing_contact_delta': 10 * TO_HOURS,
    'smart_tracing_actions': [], # any of `isolate`, `test`
    'smart_tracing_households_only': False,

    'smart_tracing_policy_isolate': None, # one of None, `basic`, `advanced`, `advanced-threshold`
    'smart_tracing_isolated_contacts': 0,
    'smart_tracing_isolation_duration': 14 * TO_HOURS,
    'smart_tracing_isolation_threshold': 1.0,

    'smart_tracing_policy_test': None,  # one of None, `basic`, `advanced`, `advanced-threshold`
    'smart_tracing_tested_contacts': 0,
    'smart_tracing_testing_threshold': 1.0,
    'trigger_tracing_after_posi_trace_test' : True,

    # beacons and manual tracing
    'p_willing_to_share': 1.0,
    'delta_manual_tracing': 1.0
}

# BO acquisition function optimization (Knowledge gradient)
# default settings from botorch
calibration_acqf = {
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
        'TR': 'LK Tirschenreuth',
        'SB': 'LK Schwarzwald-Baar-Kreis',
    },
    'CH' : {
        'SZ': 'SZ',     # Canton Schwyz
        'TI': 'TI',     # Canton Ticino
        'LU': 'LU',     # Canton Lucerne
        'BE': 'BE',     # Canton Bern
        'VD': 'VD',     # Canton Vaud
        'JU': 'JU',     # Canton Jura
        'BS': 'BS',     # Basel Stadt (Canton)
    }
}				

# lockdown dates
calibration_lockdown_dates = {
    'GER': {
        'start' : '2020-03-23',
        'end': '2020-05-03',
    },
    'CH': {
        'start': '2020-03-16',
        'end': '2020-05-10',
    },
}

# mobs settings path;
calibration_mob_paths = {
    'GER': {
        'TU': ['lib/mobility/Tubingen_settings_10.pk', 'lib/mobility/Tubingen_settings_1.pk'],
        'KL': ['lib/mobility/Kaiserslautern_settings_10.pk', 'lib/mobility/Kaiserslautern_settings_1.pk'],
        'RH': ['lib/mobility/Ruedesheim_settings_10.pk', 'lib/mobility/Ruedesheim_settings_1.pk'],
        'TR': ['lib/mobility/Tirschenreuth_settings_5.pk', 'lib/mobility/Tirschenreuth_settings_1.pk'],
        'SB': ['lib/mobility/Schwarzwald-Baar_settings_10.pk', 'lib/mobility/Schwarzwald-Baar_settings_1.pk']
    },
    'CH': {
        'VD': ['lib/mobility/Lausanne_settings_10.pk', 'lib/mobility/Lausanne_settings_1.pk'],
        'BE': ['lib/mobility/Bern_settings_10.pk', 'lib/mobility/Bern_settings_1.pk'],
        'TI': ['lib/mobility/Locarno_settings_2.pk', 'lib/mobility/Locarno_settings_1.pk'],
        'JU': ['lib/mobility/Jura_settings_10.pk', 'lib/mobility/Jura_settings_1.pk'],
        'BS': ['lib/mobility/Basel_settings_10.pk', 'lib/mobility/Basel_settings_1.pk'],
    }
}

# calibration start dates
calibration_start_dates = {
    'GER': {
        'TU': '2020-03-12',
        'KL': '2020-03-15',
        'RH': '2020-03-10',
        'TR': '2020-03-13',
        'SB': '2020-03-13',
    },
    'CH': {
        'VD': '2020-03-07',
        'BE': '2020-03-06',
        'TI': '2020-03-09',
        'JU': '2020-03-09',
        'BS': '2020-03-04',
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

# calibration lockdown beta multipliers
calibration_lockdown_beta_multipliers = {
    'education': 0.5, 
    'social': 0.5,
    'bus_stop': 0.5, 
    'office': 0.5, 
    'supermarket': 0.5}

calibration_lockdown_site_closures = ['education', 'social']
calibration_lockdown_mobility_reduced = ['supermarket', 'bus_stop', 'office']

# contact tracing adoption 
contact_tracing_adoption = {
    'GER': 0.27,  # as of Nov 12, 2020
    'CH' : 0.22,  # as of Nov 20, 2020
}

# mobility reduction
calibration_mobility_reduction = {
    'GER': {
        'SB': get_mobility_reduction('Germany', 'Baden-Württemberg', calibration_lockdown_dates['GER']['start'], calibration_lockdown_dates['GER']['end'], calibration_lockdown_mobility_reduced),
        'TU': get_mobility_reduction('Germany', 'Baden-Württemberg', calibration_lockdown_dates['GER']['start'], calibration_lockdown_dates['GER']['end'], calibration_lockdown_mobility_reduced),
        'KL': get_mobility_reduction('Germany', 'Rhineland-Palatinate', calibration_lockdown_dates['GER']['start'], calibration_lockdown_dates['GER']['end'], calibration_lockdown_mobility_reduced),
        'RH': get_mobility_reduction('Germany', 'Hessen', calibration_lockdown_dates['GER']['start'], calibration_lockdown_dates['GER']['end'], calibration_lockdown_mobility_reduced),
        'TR': get_mobility_reduction('Germany', 'Bavaria', calibration_lockdown_dates['GER']['start'], calibration_lockdown_dates['GER']['end'], calibration_lockdown_mobility_reduced),
    },
    'CH': {
        'VD': get_mobility_reduction('Switzerland', 'Vaud', calibration_lockdown_dates['CH']['start'], calibration_lockdown_dates['CH']['end'], calibration_lockdown_mobility_reduced),
        'BE': get_mobility_reduction('Switzerland', 'Canton of Bern', calibration_lockdown_dates['CH']['start'], calibration_lockdown_dates['CH']['end'], calibration_lockdown_mobility_reduced),
        'TI': get_mobility_reduction('Switzerland', 'Ticino', calibration_lockdown_dates['CH']['start'], calibration_lockdown_dates['CH']['end'], calibration_lockdown_mobility_reduced),
        'JU': get_mobility_reduction('Switzerland', 'Jura', calibration_lockdown_dates['CH']['start'], calibration_lockdown_dates['CH']['end'], calibration_lockdown_mobility_reduced),
        'BS': get_mobility_reduction('Switzerland', 'Basel City', calibration_lockdown_dates['CH']['start'], calibration_lockdown_dates['CH']['end'], calibration_lockdown_mobility_reduced),
    }
}
