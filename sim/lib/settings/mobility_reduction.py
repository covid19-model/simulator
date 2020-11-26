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
            'education' : 0.79,
            'social' : 0.46,
            'bus_stop': 0.59, 
            'office' : 0.79,
            'supermarket' : 0.86,
        },
    },
    'CH': {
        'TI': {
            'education': 0.87,
            'social' : 0.72,
            'bus_stop' : 0.77,
            'office' : 0.87,
            'supermarket' : 0.73,
        },
    },
}
