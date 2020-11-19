import numpy as np

'''
Settings for town generation
'''

town_name = 'Jura'

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path = 'lib/data/population/population_che_2019-07-01.csv'   # Population density file

sites_path='lib/data/queries/'  # Directory containing OSM site query details
bbox = (47.3000, 47.4736, 6.9514, 7.4500)   # Coordinate bounding box


# Population per age group in the region
# Source: https://www.citypopulation.de/en/switzerland/admin/26__jura/
population_per_age_group = np.array([7291,     # 0-9
                                     8104,     # 10-19
                                     9186,     # 20-29
                                     8519,     # 30-39
                                     9392,     # 40-49
                                     10891,     # 50-59
                                     8986,     # 60-69
                                     6650,     # 70-79
                                     4400       # 80+
                                     ], dtype=np.int32)

region_population = population_per_age_group.sum()
town_population = region_population     # Consider full region in simulation

# Roughly 5k tests per day in Switzerland (rough average over time frame 10.03.-27.04.2020:
# https://www.bag.admin.ch/bag/en/home/krankheiten/ausbrueche-epidemien-pandemien/aktuelle-ausbrueche-epidemien/novel-cov/situation-schweiz-und-international.html
daily_tests_unscaled = int(5000 * (town_population / 8570000))

# Information about household structure (set to None if not available)
# Source for Switzerland: https://www.bfs.admin.ch/bfs/de/home/statistiken/bevoelkerung/stand-entwicklung/haushalte.html
household_info = {
    'size_dist': [16, 29, 18, 23, 14],              # distribution of household sizes (1-5 people) in %
    'soc_role': {
        'children': [1, 8/10, 0, 0, 0, 0, 0, 0, 0],    # age groups 0,1 can be children
        'parents': [0, 2/10, 1, 1, 1, 1, 0, 0, 0],     # age groups 1,2,3,4,5 can be parents
        'elderly': [0, 0, 0, 0, 0, 0, 1, 1, 1]         # age groups 6,7,8 are elderly
    }
}
