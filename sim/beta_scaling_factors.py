"""
Example script for beta scaling factors. All model parameters are inferred by using BO on downsampled cities.
In order to adjust for the difference to full scale cities we scale the parameter 'beta' according to different heuristics.
"""

ave_total_contact_time = {
    'GER': {
        'TU': 0.9,
        'KL': 0.9,
        'RH': 0.9,
        'TR': 0.9,
    },
    'CH': {
        'VD': 0.9,
        'BE': 0.9,
        'TI': 0.9,
        'JU': 0.9,
    }
}