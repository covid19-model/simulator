import numpy as np

# -------------- Area information ----------------------------
area = 'JU'      # Area name used to name generated files, if possible set this to 'x' if file name is 'x_config.py'
area_code = 'JU'   # Code for the area of interest in the country specific case database (used by data module)
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

# Specify during which time in the calibration window lockdown measures were active in the respective area
calibration_lockdown_dates = {'start': '2020-03-16',
                              'end': '2020-05-10'}

# The calibration period should be chosen such that the corresponding area has at least x cases at the start,
# we have typically calibrated over 8-10 weeks
calibration_start_dates = '2020-03-09'
calibration_end_dates = calibration_lockdown_dates['end']

# Path of mobility settings generated with `town_generator.ipynb`, the 'downsampled' entry is used for calibration
# whereas the 'full scale' entry is used for experiments
mobility_settings = {'downscaled': f'lib/mobility/Jura_settings_10.pk',
                     'full scale': f'lib/mobility/Jura_settings_1.pk'}

# calibration states loaded for calibrated parameters
calibration_states = 'logs/calibration_ju0_state.pk'



# --------------- Country information -----------------------
country = 'CH'

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path = 'lib/data/population/population_che_2019-07-01.csv'   # Population density file

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

# Fatality and hospitalization probabilities for the specific countries for the corresponding age groups
fatality_rates_by_age = np.array([0, 0, 0, 0.001, 0.001, 0.005, 0.031, 0.111, 0.265])
p_hospital_by_age = np.array([0.155, 0.038, 0.028, 0.033, 0.054, 0.089, 0.178, 0.326, 0.29])

# Visits per week to ['education', 'social', 'bus_stop', 'office', 'supermarket'] per age group per week
# This has to be adapted to the age groups of the data for the specific country
mobility_rate_per_age_per_type = [
    [5, 1, 0, 0, 0],  # 0-9
    [5, 2, 3, 0, 0],  # 10-19
    [2, 2, 3, 3, 1],  # 20-29
    [2, 2, 3, 3, 1],  # 30-39
    [0, 2, 1, 5, 1],  # 40-49
    [0, 2, 1, 5, 1],  # 50-59
    [0, 3, 2, 0, 1],  # 60-69
    [0, 3, 2, 0, 1],  # 70-79
    [0, 2, 1, 0, 1]]  # 80+


# ---------------------- Check consistency of age groups -----------------------
assert len(household_info['soc_role']['children']) == len(population_per_age_group), 'Age groups must match.'
assert len(mobility_rate_per_age_per_type) == len(population_per_age_group), 'Age groups must match.'
assert len(fatality_rates_by_age) == len(p_hospital_by_age), 'Age groups must match.'
assert len(fatality_rates_by_age) == len(mobility_rate_per_age_per_type), 'Age groups must match.'
