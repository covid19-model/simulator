

import sys
if '..' not in sys.path:
    sys.path.append('..')
    
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import networkx as nx
import copy
import scipy as sp
import math
import seaborn
import pickle
import warnings
import os
import importlib
import argparse

from lib.mobilitysim import MobilitySimulator
from lib.town_data import generate_population, generate_sites, compute_distances
from lib.town_maps import MapIllustrator
from lib.distributions import CovidDistributions



def generate_downscaled_area(
    *, 
    downsample,
    area_name,
    country,
    population_path,
    population_per_age_group,
    region_population,
    town_population,
    daily_tests_unscaled,
    household_info,
    site_loc, 
    site_type, 
    site_dict, 
    density_site_loc,
):

    # Downsampling operations (not done at compilation time)   
    if downsample > 1:
        # population
        population_per_age_group = np.round(
            population_per_age_group * town_population / (downsample * region_population)
            ).astype('int').tolist()

        # sites
        np.random.seed(42)
        idx = np.random.choice(len(site_loc), size=int(len(site_loc) / downsample),
                            replace=False, p=np.ones(len(site_loc)) / len(site_loc))
        site_loc, site_type = np.array(site_loc)[idx].tolist(), np.array(site_type)[idx].tolist()

    print(f'Population per age group: {population_per_age_group}')
    print(f'Site types:      ', site_dict)


  
    # Generate home location based on various options
    # * `home_loc`: list of home coordinates
    # * `people_age`: list of age category
    # * `home_tile`: list of map tile to which each home belongs
    # * `tile_loc`: list tile center coordinates

    # The following three options generate a population distribution across a geographical area
    # consisting of tiles (square boxes) of specific resolution.
    # More information about tile sizes can be found in
    # https://wiki.openstreetmap.org/wiki/Zoom_levels.

    # Population density data from Facebook, e.g. for Germany:
    # https://data.humdata.org/dataset/germany-high-resolution-population-density-maps-demographic-estimates

    if region_population == town_population:
        tile_level = 15
    else:
        tile_level = 16

    if args.population_density == 'custom':
        # generate population across tiles based on density input
        print('Tile level: ', tile_level)
        home_loc, people_age, home_tile, tile_loc, people_household = generate_population(
            density_file=population_path, bbox=bbox,
            population_per_age_group=population_per_age_group,
            household_info=household_info, tile_level=tile_level, seed=42)

    elif args.population_density == 'random':
        # generate population across tiles uniformly at random
        home_loc, people_age, home_tile, tile_loc, people_household = generate_population(
            bbox=bbox, population_per_age_group=population_per_age_group,
            tile_level=16, seed=42)

    elif args.population_density == 'heuristic':
        # generate population across tiles proportional to buildings per tile
        home_loc, people_age, home_tile, tile_loc, people_household = generate_population(
            bbox=bbox, density_site_loc=density_site_loc,
            population_per_age_group=population_per_age_group, tile_level=16, seed=42)

    # Compute pairwise distances between all tile centers and all sites
    tile_site_dist = compute_distances(site_loc, tile_loc)

    # Specify synthetic mobility patterns
    # Here we specify the patterns of mobility used for generating the synthetic traces based on the above home and site locations. Note that this is a general framework and can by arbitrarilty extended to any desired site numbers or types. See below for an example used in the first version of our paper.

    # Specify the mean duration of visit per type, or in reality, time spent in crowded places per type.
    # 2h at office-education, 1.5h at restaurants/bars, 0.5 at supermarket, 0.2 at bus stop.
    dur_mean_per_type = [2, 1.5, 0.2, 2, 0.5]

    # Determine the number of discrete sites a person visits per site type.
    # 1 office, 1 school, 10 social, 2 supermarkets, 5 bus stops
    variety_per_type = [1, 10, 5, 1, 2]

    # Set the number of visits per week that each group makes per type of site
    # e.g. line 0 corresponds to age 0-4 in Germany
    # no office, a lot of education (kindergarden), some social, no supermarket, no public transport
    # the age groups are chosen to match the age groups used in case data by national authorities
    # GERMANY
    if country == 'GER':
        mob_rate_per_age_per_type = [
            [5, 1, 0, 0, 0],  # 0-4
            [5, 2, 3, 0, 0],  # 5-14
            [2, 2, 3, 3, 1],  # 15-34
            [0, 2, 1, 5, 1],  # 35-59
            [0, 3, 2, 0, 1],  # 60-79
            [0, 2, 1, 0, 1]]  # 80+

    # SWITZERLAND
    elif country == 'CH':
        mob_rate_per_age_per_type = [
            [5, 1, 0, 0, 0],  # 0-9
            [5, 2, 3, 0, 0],  # 10-19
            [2, 2, 3, 3, 1],  # 20-29
            [2, 2, 3, 3, 1],  # 30-39
            [0, 2, 1, 5, 1],  # 40-49
            [0, 2, 1, 5, 1],  # 50-59
            [0, 3, 2, 0, 1],  # 60-69
            [0, 3, 2, 0, 1],  # 70-79
            [0, 2, 1, 0, 1]]  # 80+
    else:
        raise ValueError('Invalid country code.')

    # convert to average visits per hour per week, to be compatible with simulator
    mob_rate_per_age_per_type = np.divide(
        np.array(mob_rate_per_age_per_type), (24.0 * 7))

    # Set `delta`; the setting for delta is explained in the paper.
    # time horizon
    # 4.6438 # as set by distributions
    delta = CovidDistributions(country=country).delta

    print('Population (by Age): ', population_per_age_group)
    print('Sites (by type):     ',  [
          (np.array(site_type) == i).sum() for i in range(5)])

    print('Total:', sum(population_per_age_group), len(site_type))


    # Returns arguments for the class object instantiation to be able to initiate `MobilitySimulator`
    # on the fly during calibration and for different downscaling
    print(downsample)

    kwargs = dict(
        home_loc=home_loc,
        people_age=people_age,
        num_people_unscaled=town_population,
        region_population=region_population,
        downsample=downsample,
        mob_rate_per_age_per_type=mob_rate_per_age_per_type,
        daily_tests_unscaled=daily_tests_unscaled,
        dur_mean_per_type=dur_mean_per_type,
        variety_per_type=variety_per_type,
        delta=delta,
        home_tile=home_tile,
        tile_site_dist=tile_site_dist,
        people_household=people_household,
        site_loc=site_loc,
        site_type=site_type,
        site_dict=site_dict,
        density_site_loc=density_site_loc

    )

    return kwargs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="set seed")
    parser.add_argument("-f", "--filename", help="specify filename for area compilation", required=True)
    parser.add_argument("--population-density", choices=['custom', 'random', 'heuristic'], default='custom', 
        help="specify how home locations are being generated")
    args = parser.parse_args()

    # attempt loading correct area settings
    try:
        area = importlib.import_module(args.filename)

    except ModuleNotFoundError:
        print('Settings specified by `--filename` not found.')
        exit(1)

    # explicitly load information from area settings
    try:
        area_name = area.town_name
        country = area.country
        population_path = area.population_path
        sites_path = area.sites_path
        bbox = area.bbox
        population_per_age_group = area.population_per_age_group
        region_population = area.region_population
        town_population = area.town_population
        daily_tests_unscaled = area.daily_tests_unscaled
        household_info = area.household_info
    except AttributeError:
        print('Specified area data do not contain required attributes.')
        exit(1)

    '''
    All of these do not depend on downscaling in the current pipeline
    Thus, only do this at compile time. For donwscaled area simulations, this information 
    will be used from the full scale (i.e. compiled version)
    '''

    # Extracted site data
    # * `site_loc`: list of site coordinates
    # * `site_type`: list of site category
    # * `site_dict`: helper dictionary with real name (string) of each site category (int)
    # * `density_site_loc`: list of site coordinates of specific type to be based on to generate population density

    # To generate sites of arbitrary sites for a given city, the following function sends queries to OpenStreetMap.
    # In order to use it for additional types of sites, you need to specify queries in the Overpass API format.
    # For more information, check the existing queries in
    # **/lib/data/queries/**, https://wiki.openstreetmap.org/wiki/Overpass_API and http://overpass-turbo.eu/.

    # We separatelly use a query returning all buildings in a town to heuristically generate population
    # density in the next steps if no real population density data is provided. An extra query is required
    # for this purpose and it should be given as a **site_based_density_file** argument.

    # This block sends queries to OpenStreetMap
    # Make sure you have a working internet connection
    # If an error occurs during execution, try executing again
    # If the call times out or doesn't finish, try restarting your internet connection by e.g. restarting your computer
    
    site_files = []
    for root, dirs, files in os.walk(area.sites_path):
        for f in files:
            if f.endswith(".txt") and f != 'buildings.txt':
                site_files.append(area.sites_path + f)

    site_loc, site_type, site_dict, density_site_loc = generate_sites(
        bbox=area.bbox, query_files=site_files,
        site_based_density_file='lib/data/queries/buildings.txt')

    '''
    This function depends on downscaling
    '''

    downsample = 1 # compilation

    kwargs = generate_downscaled_area(

        downsample=downsample,

        # At compile time: from static area data 
        # At downscaling time: from compiled area settings
        area_name=area_name,
        country=country,
        population_path=population_path,
        population_per_age_group=population_per_age_group,
        region_population=region_population,
        town_population=town_population,
        daily_tests_unscaled=daily_tests_unscaled,
        household_info=household_info,
        
        # At compile time: from OpenStreeMap query
        # At downscaling time: from compiled area settings
        site_loc=site_loc,
        site_type=site_type,
        site_dict=site_dict,
        density_site_loc=density_site_loc,
    )


    print(kwargs.keys())
    print('Done.')


    # with open(f'lib/mobility/{area_name}_settings_{downsample}.pk', 'wb') as fp:
    #     pickle.dump(kwargs, fp)
