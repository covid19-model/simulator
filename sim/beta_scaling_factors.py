"""
Example script for beta scaling factors. All model parameters are inferred by using BO on downsampled cities.
In order to adjust for the difference to full scale cities we scale the parameter 'beta' according to different heuristics.
"""

ave_total_contact_time = {
    'GER': {
        'TU': 1.0224662845761705,
        'KL': 0.9004104940742766,
        'RH': 0.8753245679469659,
        'TR': 0.8459067684786509,
    },
    'CH': {
        'VD': 0.8766019032439648,
        'BE': 0.8826423909034349,
        'TI': 0.9994644931716655,
        'JU': 1.0779199179228354,
    }
}

ave_ave_contact_time = {
    'GER': {
        'TU': 0.9460325623588564,
        'KL': 1.0131156132119097,
        'RH': 0.9734140102874856,
        'TR': 0.9827709124381999,
    },
    'CH': {
        'VD': 1.0007869092312083,
        'BE': 1.0027094336750433,
        'TI': 0.9898882760715642,
        'JU': 1.0124625403797970,
    }
}

ave_ave_contact_time_unique = {
    'GER': {
        'TU': 0.9485979452888926,
        'KL': 1.0265845105516123,
        'RH': 0.9740349199798185,
        'TR': 0.9900679952384168,
    },
    'CH': {
        'VD': 1.0079881266781230,
        'BE': 1.0063074680755202,
        'TI': 0.9886354492545409,
        'JU': 1.0324624410810717,
    }
}
