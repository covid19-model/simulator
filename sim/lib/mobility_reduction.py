import numpy as np
import pandas as pd

'''
Settings for mobility reduction

Google COVID19 mobility reports:

Jan 22 2021
https://www.google.com/covid19/mobility/

'''

mobility_change_keywords = {
    'education': 'workplaces_percent_change_from_baseline',
    'social': 'retail_and_recreation_percent_change_from_baseline',
    'bus_stop': 'transit_stations_percent_change_from_baseline',
    'office': 'workplaces_percent_change_from_baseline',
    'supermarket': 'grocery_and_pharmacy_percent_change_from_baseline',
}

def get_mobility_reduction(country, region, start_date, end_date, site_type_list):

    # get country dataset
    if country == 'Germany':
        df = pd.read_csv('lib/data/mobility/2020_DE_Region_Mobility_Report.csv', header=0, delimiter=',')
    elif country == 'Switzerland':
        df = pd.read_csv('lib/data/mobility/2020_CH_Region_Mobility_Report.csv', header=0, delimiter=',')
    else:
        raise KeyError('Invalid country for mobility reduction data.')

    # filter region
    df = df[df['sub_region_1'] == region]

    # filter dates
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # average mobility reduction
    mean_mobility_reduction_dict = {
        k: max(0.0, - df[[v]].values.mean() / 100.0) # from perc. reduction to prob. of missing visit
        for k, v in mobility_change_keywords.items() if k in site_type_list
    }

    return mean_mobility_reduction_dict
     
