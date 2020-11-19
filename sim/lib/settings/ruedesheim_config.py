import numpy as np

# -------------- Area information ----------------------------
area = 'RH'      # Area name used to name generated files, if possible set this to 'x' if file name is 'x_config.py'
area_code = 'LK Rheingau-Taunus-Kreis'   # Code for the area of interest in the country specific case database (used by data module)
bbox = (49.9680, 50.2823, 7.7715, 8.1752)   # Coordinate bounding box

# Population per age group in the region (matching the RKI age groups)
# Source for Germany: https://www.citypopulation.de/en/germany/
population_per_age_group = np.array([
    8150,   # 0-4
    17265,  # 5-14
    38489,  # 15-34
    67634,  # 35-59
    43309,  # 60-79
    12309,  # 80+
    ], dtype=np.int32)

region_population = population_per_age_group.sum()
town_population = region_population

# Specify during which time in the calibration window lockdown measures were active in the respective area
calibration_lockdown_dates = {'start': '2020-03-23',
                              'end': '2020-05-03'}

# The calibration period should be chosen such that the corresponding area has at least x cases at the start,
# we have typically calibrated over 8-10 weeks
calibration_start_dates = '2020-03-10'
calibration_end_dates = calibration_lockdown_dates['end']

# Path of mobility settings generated with `town_generator.ipynb`, the 'downsampled' entry is used for calibration
# whereas the 'full scale' entry is used for experiments
mobility_settings = {'downscaled': f'lib/mobility/Ruedesheim_settings_10.pk',
                     'full scale': f'lib/mobility/Ruedesheim_settings_1.pk'}

# calibration states loaded for calibrated parameters
calibration_states = 'logs/calibration_rh0_state.pk'



# --------------- Country information -----------------------
country = 'GER'

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path = 'lib/data/population/population_deu_2019-07-01.csv'

# Information about household structure (set to None if not available)
# Source for Germany: https://www.destatis.de/EN/Themes/Society-Environment/Population/Households-Families/Tables/lrbev05.html
household_info = {
    'size_dist': [41.9, 33.8, 11.9, 9.1, 3.4],  # distribution of household sizes (1-5 people)
    'soc_role': {
        'children': [1, 1, 3/20, 0, 0, 0],  # age groups 0,1,2 can be children
        'parents': [0, 0, 17/20, 1, 0, 0],  # age groups 2,3 can be parents
        'elderly': [0, 0, 0, 0, 1, 1]   # age groups 4,5 are elderly
    }
}

# Fatality and hospitalization probabilities for the specific countries for the corresponding age groups
fatality_rates_by_age = np.array([0.0, 0.0, 0.0, 0.004, 0.073, 0.247])
p_hospital_by_age = np.array([0.001, 0.002, 0.012, 0.065, 0.205, 0.273])

# Visits per week to ['education', 'social', 'bus_stop', 'office', 'supermarket'] per age group per week
# This has to be adapted to the age groups of the data for the specific country
mobility_rate_per_age_per_type = [
    [5, 1, 0, 0, 0],  # 0-4
    [5, 2, 3, 0, 0],  # 5-14
    [2, 2, 3, 3, 1],  # 15-34
    [0, 2, 1, 5, 1],  # 35-59
    [0, 3, 2, 0, 1],  # 60-79
    [0, 2, 1, 0, 1]]  # 80+


# ---------------------- Check consistency of age groups -----------------------
assert len(household_info['soc_role']['children']) == len(population_per_age_group), 'Age groups must match.'
assert len(mobility_rate_per_age_per_type) == len(population_per_age_group), 'Age groups must match.'
assert len(fatality_rates_by_age) == len(p_hospital_by_age), 'Age groups must match.'
assert len(fatality_rates_by_age) == len(mobility_rate_per_age_per_type), 'Age groups must match.'
