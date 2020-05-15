import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import geopy.distance
import requests

TO_HOURS = 24.0

# tile levels and corresponding width (degrees of longitudes)
# from OpenStreetMap (https://wiki.openstreetmap.org/wiki/Zoom_levels) 
tile_level_dict = {
    0: 360,
    1: 180,
    2: 90,
    3: 45,
    4: 22.5,
    5: 11.25,
    6: 5.625,
    7: 2.813,
    8: 1.406,
    9: 0.703,
    10: 0.352,
    11: 0.176,
    12: 0.088,
    13: 0.044,
    14: 0.022,
    15: 0.011,
    16: 0.005,
    17: 0.003,
    18: 0.001,
    19: 0.0005,
    20: 0.00025
}

def generate_population(bbox, population_per_age_group, density_file=None, tile_level=16, seed=None,
                        density_site_loc=None, household_info=None):
    
    # raise error if tile level is invalid
    assert (type(tile_level)==int and tile_level>=0 and tile_level<=20), 'Invalid tile level'

    # input seed for reproducibility
    if seed is not None:
        np.random.seed(seed=seed)

    # tile size in degrees
    tile_size = tile_level_dict[tile_level] 

    # total population
    population = sum(population_per_age_group)
    
    if density_file is not None:

        # read population density file
        pops = pd.read_csv(density_file)

        # discard records out of the bounding box
        pops = pops.loc[(pops['Lat'] >= bbox[0]) & (pops['Lat'] <= bbox[1]) & (pops['Lon'] >= bbox[2]) & (pops['Lon'] <= bbox[3])]
        
        # split the map into rectangular tiles
        lat_arr = np.arange(bbox[0]+tile_size/2, bbox[1]-tile_size/2, tile_size)
        lon_arr = np.arange(bbox[2]+tile_size/2, bbox[3]-tile_size/2, tile_size)
        num_of_tiles = len(lat_arr)*len(lon_arr)

        tiles = pd.DataFrame()
        for lat in lat_arr:
            for lon in lon_arr:
                # compute the total population records in each tile
                pops_in_tile = pops.loc[(pops['Lat'] >= lat-tile_size/2) & (pops['Lat'] <= lat+tile_size/2) & (pops['Lon'] >= lon-tile_size/2) & (pops['Lon'] <= lon+tile_size/2)]
                tiles = tiles.append(pd.DataFrame(data={'lat': [lat], 'lon': [lon], 'pop': [sum(pops_in_tile['Population'])]}))

        # scale population density to real numbers
        tiles['pop'] /= sum(tiles['pop'])
        tiles['pop'] *= population
        tiles['pop'] = tiles['pop'].round().astype(int)

    elif density_file is None and density_site_loc is None:

        # generate a grid of tiles inside the bounding box
        lat_arr = np.arange(bbox[0]+tile_size/2, bbox[1]-tile_size/2, tile_size)
        lon_arr = np.arange(bbox[2]+tile_size/2, bbox[3]-tile_size/2, tile_size)
        num_of_tiles = len(lat_arr)*len(lon_arr)

        # set probabilities proportional to density 
        density_prob = num_of_tiles*[1/num_of_tiles]
        # generate population equally distributed accross all tiles 
        population_distribution = np.random.multinomial(population, density_prob, size=1)[0]
        
        tiles=pd.DataFrame()
        tile_ind=0
        for lat in lat_arr:
            for lon in lon_arr:
                tiles = tiles.append(pd.DataFrame(data={'lat': [lat], 'lon': [lon], 'pop': [population_distribution[tile_ind]]}))
                tile_ind += 1
        
    elif density_file is None and density_site_loc is not None:

        # generate a grid of tiles inside the bounding box
        lat_arr = np.arange(bbox[0]+tile_size/2, bbox[1]-tile_size/2, tile_size)
        lon_arr = np.arange(bbox[2]+tile_size/2, bbox[3]-tile_size/2, tile_size)
        num_of_tiles = len(lat_arr)*len(lon_arr)

        num_critical_sites = len(density_site_loc)

        # set probabilities proportional to density 
        density_prob = num_of_tiles*[0]
        
        tiles=pd.DataFrame()
        tile_ind=0
        for lat in lat_arr:
            for lon in lon_arr:
                num_critical_sites_in_tile=0
                for site_lat, site_lon in density_site_loc:
                    if site_lat>=lat-tile_size/2 and site_lat<=lat+tile_size/2 and site_lon>=lon-tile_size/2 and site_lon<=lon+tile_size/2:
                        num_critical_sites_in_tile += 1
                density_prob[tile_ind] = num_critical_sites_in_tile/num_critical_sites
                tile_ind += 1
        

        # generate population proportional to the critical sites per tile (e.g. bus stops) 
        population_distribution = np.random.multinomial(population, density_prob, size=1)[0]

        tile_ind=0
        for lat in lat_arr:
            for lon in lon_arr:
                tiles = tiles.append(pd.DataFrame(data={'lat': [lat], 'lon': [lon], 'pop': [population_distribution[tile_ind]]}))
                tile_ind += 1
    
    # discard tiles with zero population
    tiles = tiles[tiles['pop']!=0]

    # probability of being in each age group
    age_proportions = np.divide(population_per_age_group, sum(population_per_age_group))

    # generate lists of individuals' home location and age group
    home_loc=[]
    people_age=[]
    home_tile=[]
    tile_loc=[]
    i_tile=0
    for _, t in tiles.iterrows():
        lat=t['lat']
        lon=t['lon']
        pop=int(t['pop'])
        # store the coordinates of the tile center
        tile_loc.append([lat, lon])
        # generate random home locations within the tile
        home_lat = lat + tile_size*(np.random.rand(pop)-0.5)
        home_lon = lon + tile_size*(np.random.rand(pop)-0.5)
        home_loc += [[lat,lon] for lat,lon in zip(home_lat, home_lon)]
        # store the tile to which each home belongs
        home_tile+=pop*[i_tile]
        # age group assigned proportionally to the real statistics
        people_age+=list(np.random.multinomial(n=1, pvals=age_proportions, size=pop).argmax(axis=1))
        i_tile+=1

    if household_info is not None:
        # pick a societal role for each person depending on the age group
        children = 0
        soc_role = []
        for i_person, age_group in enumerate(people_age):
            soc_role_distribution = [household_info['soc_role']['children'][age_group],
                                    household_info['soc_role']['parents'][age_group],
                                    household_info['soc_role']['elderly'][age_group]]
            soc_role.append(np.random.multinomial(n=1, pvals=soc_role_distribution, size=1).argmax(axis=1)[0])
            
            if soc_role[i_person] == 0:
                children += 1

        soc_role = np.array(soc_role)
        
        household_index = 0
        people_household = np.full(len(home_loc), -1)


        # percentage of households with more than 2 people
        percent_with_children = sum(household_info['size_dist'][2:])
        # number of households with more than 2 people assuming the extra people are children
        households_with_children = children/sum([(ind+1)*perc/percent_with_children for ind, perc in enumerate(household_info['size_dist'][2:])])

        for ind, perc in enumerate(household_info['size_dist'][2:]):
            # percentage of families with ind+1 children compared to total families with children
            relative_perc = perc/percent_with_children
            family_children = ind+1
            # number of families with ind+1 children
            num_of_households = int(relative_perc*households_with_children)
            
            for _ in range(num_of_households):
                # find candidate parents and children
                candidate_parents = np.where(np.logical_and(people_household == -1, soc_role == 1))[0]
                candidate_children = np.where(np.logical_and(people_household == -1, soc_role == 0))[0]
                
                # check if the parents and children are enough
                if len(candidate_parents)>=2 and len(candidate_children)>=family_children:
                    children_to_chose = family_children
                elif len(candidate_parents)>=2 and len(candidate_children)>0:
                    children_to_chose = len(candidate_children)
                else:
                    break

                # randomly pick 2 parents and the respective children
                parents_ids = np.random.choice(candidate_parents, 2, replace=False)
                children_ids = np.random.choice(candidate_children, children_to_chose, replace=False)
                
                # store the common household number
                people_household[parents_ids] = household_index
                people_household[children_ids] = household_index
                household_index += 1

                # pick one family member and set its home location as every member's home 
                home_owner = np.random.choice(np.concatenate([parents_ids,children_ids]), 1)[0]
                for i_person in np.concatenate([parents_ids,children_ids]):
                    home_loc[i_person] = home_loc[home_owner]
                    home_tile[i_person] = home_tile[home_owner] 

        # percentage of households with 1 or 2 people
        percent_without_children = sum(household_info['size_dist'][:2])
        # people not assigned yet
        remaining = len(np.where(people_household == -1)[0])
        # number of households with 1 or 2 people
        households_with_couples = int((household_info['size_dist'][1]/percent_without_children)*remaining/((household_info['size_dist'][0]+2*household_info['size_dist'][1])/percent_without_children))

        for _ in range(households_with_couples):
                # find candidate elderly people to form a couple
                candidate_couple = np.where(np.logical_and(people_household == -1, soc_role == 2))[0]
                # if elderly people are not enough form a couple using younger people
                if len(candidate_couple)<2:
                    candidate_couple = np.where(np.logical_and(people_household == -1, soc_role == 1))[0]
                # check if a couple can be formed at all
                if len(candidate_couple)<2:
                    break
                
                # randomly pick 2 people to form a couple
                couple_ids = np.random.choice(candidate_couple, 2, replace=False)
                # store the common household number
                people_household[couple_ids] = household_index
                household_index += 1
                
                # pick one family member and set its home location as every member's home
                home_owner = np.random.choice(couple_ids, 1)[0]
                for i_person in couple_ids:
                    home_loc[i_person] = home_loc[home_owner]
                    home_tile[i_person] = home_tile[home_owner]
                
        # set all remaining people as independent 1-person families
        for i_person, family in enumerate(people_household):
            if family == -1:
                people_household[i_person] = household_index
                household_index += 1
        
    else:
        # set all people as independent 1-person families
        people_household = np.array([i for i in range(len(home_loc))])

    return home_loc, people_age, home_tile, tile_loc, people_household

def overpass_query(bbox, contents):
    overpass_bbox = str((bbox[0],bbox[2],bbox[1],bbox[3]))
    query = '[out:json][timeout:2500];('
    for x in contents:
        query += str(x)+str(overpass_bbox)+';'
    query += '); out center;'
    return query

def generate_sites(bbox, query_files, site_based_density_file=None):
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    site_loc=[]
    site_type=[]
    site_dict={}
    density_site_loc=[]

    type_ind=0
    for q_ind, qf in enumerate(query_files):
        with open(qf, 'r') as q:

            # site type is extracted by the txt file name
            s_type = qf.split('/')[-1].replace('.txt','')

            # site type index and actual name correspondence
            site_dict[type_ind]=s_type

            # read all query parameters
            contents = q.readlines()
            contents = [c for c in contents if c!='']

            # generate and call overpass queries 
            response = requests.get(overpass_url, params={'data': overpass_query(bbox, contents)})
            if response.status_code == 200:
                print('Query ' + str(q_ind+1) + ' OK.')
            else:
                print('Query ' + str(q_ind+1) + ' returned http code ' + str(response.status_code) + '. Try again.')
                return None, None, None, None
            data = response.json()

            # read sites latitude and longitude
            locs_to_add=[]
            for site in data['elements']:
                if site['type']=='way':
                    locs_to_add.append([site['center']['lat'], site['center']['lon']])
                elif site['type']=='node':
                    locs_to_add.append([site['lat'], site['lon']])

            site_type += len(locs_to_add)*[type_ind]
            site_loc += locs_to_add
            type_ind+=1
            
    # locations of this type are used to generate population density
    if site_based_density_file is not None:
        
        with open(site_based_density_file, 'r') as q:
            
            # read all query parameters
            contents = q.readlines()
            contents = [c for c in contents if c!='']

            # generate and call overpass queries 
            response = requests.get(overpass_url, params={'data': overpass_query(bbox, contents)})
            if response.status_code == 200:
                print('Query ' + str(len(query_files)+1) + ' OK.')
            else:
                print('Query ' + str(len(query_files)+1) + ' returned http code ' + str(response.status_code) + '. Try again.')
                return None, None, None, None
            data = response.json()
            
            # read sites latitude and longitude
            density_site_loc=[]
            for site in data['elements']:
                if site['type']=='way' or site['type']=='relation':
                    density_site_loc.append([site['center']['lat'], site['center']['lon']])
                elif site['type']=='node':
                    density_site_loc.append([site['lat'], site['lon']])
            

    return site_loc, site_type, site_dict, density_site_loc

def compute_distances(site_loc, tile_loc):
    
    # 2D array containing pairwise distances
    tile_site_dist=np.zeros((len(tile_loc), len(site_loc)))
    
    for i_tile, tile in enumerate(tile_loc):
        for i_site, site in enumerate(site_loc):
            tile_site_dist[i_tile,i_site]=geopy.distance.distance(tile,site).km

    return tile_site_dist
