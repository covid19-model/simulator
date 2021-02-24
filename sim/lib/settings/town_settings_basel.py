import numpy as np

'''
Settings for town generation
'''

town_name = 'Basel'
country = 'CH' 

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path = 'lib/data/population/population_che_2019-07-01.csv'   # Population density file

sites_path='lib/data/queries/'  # Directory containing OSM site query details
bbox = (47.536010, 47.576287, 7.561370, 7.624698)   # Coordinate bounding box


# Population per age group in the region
# Source: https://www.citypopulation.de/en/switzerland/admin/basel_stadt/B1200__basel_stadt/

population_per_age_group = np.array([18565,     # 0-9
                                     15404,     # 10-19
                                     23685,     # 20-29
                                     33539,     # 30-39
                                     26859,     # 40-49
                                     27696,     # 50-59
                                     20767,     # 60-69
                                     16111,     # 70-79
                                     13218,     # 80+
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
