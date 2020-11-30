import numpy as np

'''
Settings for mobility reduction

Google COVID19 mobility reports:

Nov 11 2020
https://www.gstatic.com/covid19/mobility/2020-11-22_DE_Mobility_Report_en-GB.pdf
https://www.gstatic.com/covid19/mobility/2020-11-22_CH_Mobility_Report_en.pdf

'''

mobility_reduction = {
    'GER' : {
        'TU' : {
            'education' : 0.21,
            'social' : 0.54,
            'bus_stop': 0.41, 
            'office' : 0.21,
            'supermarket' : 0.14,
        },
    },
    'CH': {
        'TI': {
            'education': 0.13,
            'social' : 0.28,
            'bus_stop' : 0.23,
            'office' : 0.13,
            'supermarket' : 0.27,
        },
    },
}
