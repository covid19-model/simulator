from lib.mobilitysim import compute_mean_invariant_beta_multipliers


TO_HOURS = 24.0

invariant_beta_multipliers = {
    'GER': {
        'TU': {
            1: {
                'education': 1.0,
                'social': 1.0,
                'bus_stop': 1.0,
                'office': 1.0,
                'supermarket': 1.0,
                },
            2: {'education': 0.6975848532586864,
                'social': 1.3951697065173727,
                'bus_stop': 0.3487924266293432,
                'office': 0.6975848532586864,
                'supermarket': 0.6975848532586864
                },
            5: {'education': 0.36478840278856123,
                'social': 1.8239420139428062,
                'bus_stop': 0.07295768055771225,
                'office': 0.36478840278856123,
                'supermarket': 0.36478840278856123
                },
            10: {'education': 0.20348036231227085,
                 'social': 2.0348036231227082,
                 'bus_stop': 0.020348036231227086,
                 'office': 0.20348036231227085,
                 'supermarket': 0.20348036231227085
                },
            },
        },
    }


def get_invariant_beta_multiplier(dispersion_factor, country, area, use_invariant_rescaling=True, verbose=True):
    try:
        beta_multipliers = invariant_beta_multipliers[country][area][dispersion_factor]
    except KeyError:
        if verbose:
            print('Requested beta multipliers not yet available. Calculate invariant beta multipliers ...')

        beta_multipliers = {
            'education': 1.0,
            'social': 1.0 * dispersion_factor,
            'bus_stop': 1.0 / dispersion_factor,
            'office': 1.0,
            'supermarket': 1.0,
        }
        if use_invariant_rescaling:
            beta_multipliers = compute_mean_invariant_beta_multipliers(beta_multipliers=beta_multipliers,
                                                                       country=country, area=area,
                                                                       max_time=28 * TO_HOURS,
                                                                       full_scale=True,
                                                                       weighting='integrated_contact_time',
                                                                       mode='rescale_all')
    if verbose:
        print(f'Using multipliers: {beta_multipliers}')
    return beta_multipliers
