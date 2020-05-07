import sys
import argparse
# if '..' not in sys.path:
#     sys.path.append('..')

import time
import bisect
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
import scipy as sp
import os
import matplotlib.pyplot as plt
import random

from lib.settings.calibration_settings import command_line_area_codes


def get_preprocessed_data_germany(landkreis='LK Tübingen', start_date_string='2020-03-10', until=17):
    '''
    Preprocesses data for a specific Landkreis in Germany
    Data taken from
    https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0?orderBy=Bundesland

    List of Landkreis districts in `data_availability_GER.md`

    '''

    # preprocessing
    df = pd.read_csv('lib/data/RKI_COVID19.csv', header=0, delimiter=',')
    # print('Data last updated at: ', df.Datenstand.unique()[0])

    # delete unnecessary
    df = df[df['Landkreis'] == landkreis]
    df.drop(['Datenstand', 'IdLandkreis', 'Refdatum', 'ObjectId',
            'Landkreis', 'IdBundesland', 'Bundesland', 'Geschlecht'], axis=1, inplace=True)

    # delete weird data rows (insignificant)
    df = df[df.Altersgruppe != 'unbekannt'] # this is just 1 row

    # Altersgruppe map
    agemap = {
        'A00-A04' : 0,
        'A05-A14' : 1,
        'A15-A34' : 2,
        'A35-A59' : 3,
        'A60-A79' : 4,
        'A80+' : 5,
    }
    df['age_group'] = 0
    for k,v in agemap.items():
        df.loc[df.Altersgruppe == k, 'age_group'] = v
    df.drop(['Altersgruppe'], axis=1, inplace=True)

    # process date to a number of days until start of actual case growth
    df.Meldedatum = pd.to_datetime(df.Meldedatum)
    start_date = pd.to_datetime(start_date_string + ' 00:00:00+00:00') # only 4 cases in 2 weeks before that

    # discard earlier cases for simplicity
    df['days'] = (df.Meldedatum - start_date).dt.days
    df['Meldedatum'] = df.Meldedatum.dt.date
    df = df[df['days'] >= 0]

    # filter days after March 26 (measures started at March 23, plus lag and incubation time)
    df = df[df['days'] <= until] # until = 16 in inference

    return df


def collect_data_from_df(country, area, datatype, start_date_string, days):
    '''
    Collects data for a country `country` and a specific area `area` for `days` days, 
    either: new, recovered, fatality cases from df 

    `datatype` has to be one of `new`, `recovered`, `fatality`

    Returns np.array of shape (`max_days`, age_groups)
    '''

    if country == 'GER':

        if datatype == 'new':
            ctr, indic = 'AnzahlFall', 'NeuerFall'
        elif datatype == 'recovered':
            ctr, indic = 'AnzahlGenesen', 'NeuGenesen'
        elif datatype == 'fatality':
            ctr, indic = 'AnzahlTodesfall', 'NeuerTodesfall'
        else:
            raise ValueError('Invalid datatype requested.')

        if area in command_line_area_codes.keys():
            landkreis = command_line_area_codes[area]
        else:
            raise ValueError('Invalid Landkreis requested.')

        df_tmp = get_preprocessed_data_germany(landkreis=landkreis, start_date_string=start_date_string, until=days)

        # check whether the new case counts, i.e. wasn't used in a different publication
        counts_as_new = np.array((df_tmp[indic] == 0) | (df_tmp[indic] == 1), dtype='int')

        df_tmp['new'] = counts_as_new * df_tmp[ctr]

        # count up each day and them make cumulative
        # x_labels = []
        maxt = df_tmp.days.max() + 1
        data = np.zeros((maxt, 6)) # value, agegroup
        for t in range(maxt):
            # x_labels.append(df_tmp[df_tmp.days == t].Meldedatum.iloc[0].strftime('%B %d, %Y')) #"%Y-%m-%d"
            for agegroup in range(6):
                data[t, agegroup] += df_tmp[
                    (df_tmp.days == t) & (df_tmp.age_group == agegroup)].new.sum()
                
            # make cumulative
            if t > 0:
                data[t, :] += data[t - 1, :]
            
        # plt.plot(np.linspace(0, maxt-1, num=maxt), data.sum(axis=1))
        return data

    else:
        raise NotImplementedError('Invalid country requested.')
        
