import numpy as np

'''
Settings for town generation
'''

town_name = 'Lucerne'

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path = 'lib/data/population/population_che_2019-07-01.csv'   # Population density file

sites_path='lib/data/queries/'  # Directory containing OSM site query details
# FIXME: Need to add bbox, bbox = (49.4096, 49.4633, 7.6877, 7.8147) # Coordinate bounding box

# Population per age group in the region
# Source: https://www.citypopulation.de/en/switzerland/admin/03__luzern/
population_per_age_group = np.array([42599,     # 0-9
                                     40764,     # 10-19
                                     54386,     # 20-29
                                     58107,     # 30-39
                                     55842,     # 40-49
                                     61585,     # 50-59
                                     44203,     # 60-69
                                     31463,     # 70-79
                                     20608       # 80+
                                     ])

region_population = population_per_age_group.sum()
town_population = 81673     # Consider full region in simulation

# Daily testing capacity per 100k people
# Roughly 100k tests per day in Germany: https://www.rki.de/DE/Content/Infekt/EpidBull/Archiv/2020/Ausgaben/15_20.pdf?__blob=publicationFile
daily_tests_per_100k = int(100000 * town_population / 83000000)

# Information about household structure (set to None if not available)
# Source for Switzerland: https://www.bfs.admin.ch/bfs/de/home/statistiken/bevoelkerung/stand-entwicklung/haushalte.html
household_info = {
    'size_dist': [16, 29, 18, 23, 14],              # distribution of household sizes (1-5 people) in %
    'soc_role': {
        'children': [1, 8/10, 0, 0, 0, 0, 0, 0],    # age groups 0,1 can be children
        'parents': [0, 2/10, 1, 1, 1, 1, 0, 0],     # age groups 1,2,3,4,5,6 can be parents
        'elderly': [0, 0, 0, 0, 0, 1, 1, 1]         # age groups 6,7,8 are elderly
    }
}
