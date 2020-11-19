import numpy as np

# -------------- Area information ----------------------------
area = 'barcelona'      # Area name used to name generated files, if possible set this to 'x' if file name is 'x_config.py'
area_code = 'B'   # Code for the area of interest in the country specific case database (used by data module)
bbox = (41.353085, 41.449772, 2.117202, 2.222592)   # Coordinate bounding box

# Population per age group in the region
# Source: https://www.citypopulation.de/en/switzerland/admin/02__bern/
population_per_age_group = np.array([136248,     # 0-9
                                     135295,      # 10-19
                                     188708,     # 20-29
                                     250374,     # 30-39
                                     252577,     # 40-49
                                     216627,     # 50-59
                                     176926,     # 60-69
                                     138538,      # 70-79
                                     125050       # 80+
                                     ], dtype=np.int32)

region_population = population_per_age_group.sum()
town_population = 1620342

# Specify during which time in the calibration window lockdown measures were active in the respective area
# Source: https://en.wikipedia.org/wiki/COVID-19_pandemic_in_Spain
calibration_lockdown_dates = {'start': '2020-03-14',
                              'end': '2020-05-09'}

# The calibration period should be chosen such that the corresponding area has at least x cases at the start,
# we have typically calibrated over 8-10 weeks
calibration_start_dates = '2020-03-01'
calibration_end_dates = calibration_lockdown_dates['end']

# Path of mobility settings generated with `town_generator.ipynb`, the 'downsampled' entry is used for calibration
# whereas the 'full scale' entry is used for experiments
mobility_settings = {'downscaled': f'lib/mobility/{area}_settings_50.pk',
                     'full scale': f'lib/mobility/{area}_settings_50.pk'}

# calibration states loaded for calibrated parameters
calibration_states = f'logs/calibration_{area}_state.pk'



# --------------- Country information -----------------------
country = 'spain'

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path = 'lib/data/population/population_esp_2019-07-01.csv'   # Population density file

# Information about household structure (set to None if not available)
# Source for Spain: https://www.ine.es/en/prensa/ech_2017_en.pdf
household_info = {
    'size_dist': [25.4, 30.4, 20.9, 17.6, 5.7],              # distribution of household sizes (1-5 people) in %
    'soc_role': {
        'children': [1, 8/10, 0, 0, 0, 0, 0, 0, 0],    # age groups 0,1 can be children
        'parents': [0, 2/10, 1, 1, 1, 1, 0, 0, 0],     # age groups 1,2,3,4,5 can be parents
        'elderly': [0, 0, 0, 0, 0, 0, 1, 1, 1]         # age groups 6,7,8 are elderly
    }
}

# Fatality and hospitalization probabilities for the specific countries for the corresponding age groups
# Source: https://www.statista.com/statistics/1105596/covid-19-mortality-rate-by-age-group-in-spain-march/
fatality_rates_by_age = np.array([0.002, 0.003, 0.002, 0.003, 0.006, 0.015, 0.051, 0.145, 0.217])
# Source: https://www.statista.com/statistics/1106425/covid-19-mortality-rate-by-age-group-in-spain-march/
p_hospital_by_age = np.array([0.003, 0.003, 0.016, 0.041, 0.096, 0.156, 0.194, 0.232, 0.198])

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
