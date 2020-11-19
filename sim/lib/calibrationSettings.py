import multiprocessing

'''
Default settings for model calibration
'''

TO_HOURS = 24.0

calibration_data = {
    'verbose': True,
    'use_households': True,
    'data_start_date': '2020-03-10',
}

calibration_simulation = {
    'n_init_samples': 20,   # initial random evaluations
    'n_iterations': 80,     # iterations of BO
    'simulation_roll_outs': 96,     # roll-outs done in parallel per parameter setting
    'cpu_count':  multiprocessing.cpu_count(),  # cpus used for parallel computation
    'lazy_contacts': True,
}

# parameter bounds
# beta_upper_bound = 2.0    # for regular simulations
beta_upper_bound = 1.5  # for lighly affected simulations

calibration_model_param_bounds_single = {
    'beta_site': [0.0, beta_upper_bound],
    'beta_household': [0.0, beta_upper_bound],
    'p_stay_home': [0.0, 1.0],
}

calibration_model_param_bounds_multi = {
    'betas': {
        'education': [0.0, beta_upper_bound],
        'social': [0.0, beta_upper_bound],
        'bus_stop': [0.0, beta_upper_bound],
        'office': [0.0, beta_upper_bound],
        'supermarket': [0.0, beta_upper_bound],
    },
    'beta_household': [0.0, beta_upper_bound],
    'p_stay_home': [0.0, 1.0],
}


# set testing parameters
calibration_testing_params = {
    'testing_t_window': None,  # [set automatically in code]
    'testing_frequency': 1 * TO_HOURS,
    'test_reporting_lag': 2 * TO_HOURS,
    'tests_per_batch': None,  # [set automatically in code]
    'test_fpr': 0.0,
    'test_fnr': 0.0,
    'test_targets': 'isym',
    'test_queue_policy': 'fifo',

    # smart tracing
    'smart_tracing_contact_delta': 10 * TO_HOURS,
    'smart_tracing_actions': [], # any of `isolate`, `test`

    'smart_tracing_policy_isolate': None, # one of None, `basic`, `advanced`
    'smart_tracing_isolated_contacts': 0,
    'smart_tracing_isolation_duration': 14 * TO_HOURS,

    'smart_tracing_policy_test': None,  # one of None, `basic`, `advanced`
    'smart_tracing_tested_contacts': 0,
}

# BO acquisition function optimization (Knowledge gradient)
# default settings from botorch
calibration_acqf = {
    'acqf_opt_num_fantasies': 64,
    'acqf_opt_num_restarts': 10,
    'acqf_opt_raw_samples': 512,
    'acqf_opt_batch_limit': 5,
    'acqf_opt_maxiter': 20,
}


# calibration lockdown beta multipliers
calibration_lockdown_beta_multipliers = {
    'education': 0.5,
    'social': 0.5,
    'bus_stop': 1.0,
    'office': 0.5,
    'supermarket': 1.0}