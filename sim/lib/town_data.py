import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import geopy.distance
import requests

# tile levels and corresponding width (degrees of longitudes)
# from OpenStreetMaps (https://wiki.openstreetmap.org/wiki/Zoom_levels) 
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

def generate_population(bbox, population_per_age_group, density_files=None, tile_level=16, seed=None, density_site_loc=None):
    
    # raise error if tile level is invalid
    assert (type(tile_level)==int and tile_level>=0 and tile_level<=20), 'Invalid tile level'

    # input seed for reproducibility
    if seed is not None:
        np.random.seed(seed=seed)

    # tile size in degrees
    tile_size = tile_level_dict[tile_level] 

    # total population
    population = sum(population_per_age_group)
    
    if density_files is not None:

        tiles=pd.DataFrame()
        for f in density_files:
            df = pd.read_csv(f)
            # read population files and select baseline density per tile
            tiles = tiles.append(df[['lat','lon','Baseline: People']])

        tiles = tiles.rename(columns={"Baseline: People": "pop"})
        tiles = tiles.dropna(axis=0, how='any')
        
        # average over all days
        tiles = tiles.groupby(['lat','lon'], as_index=False).mean()

        # discard tiles out of the bounding box
        tiles = tiles.loc[(tiles['lat'] >= bbox[0]) & (tiles['lat'] <= bbox[1]) & (tiles['lon'] >= bbox[2]) & (tiles['lon'] <= bbox[3])]

        # scale population density to real numbers
        tiles['pop'] /= sum(tiles['pop'])
        tiles['pop'] *= population
        tiles['pop'] = tiles['pop'].round().astype(int)

    elif density_files is None and density_site_loc is None:

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
        
    elif density_files is None and density_site_loc is not None:

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
    
    return home_loc, people_age, home_tile, tile_loc

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
    for qf in query_files:
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
            data = response.json()

            # read sites latitude and longitude
            ways=[]
            nodes=[]
            for site in data['elements']:
                if site['type']=='way':
                    ways.append([site['center']['lat'], site['center']['lon']])
                elif site['type']=='node':
                    nodes.append([site['lat'], site['lon']])
            
            # if there are ways in the query results discard all nodes
            if ways==[] and nodes!=[]:
                locs_to_add = nodes
                site_type += len(nodes)*[type_ind]
            elif ways!=[]:
                locs_to_add = ways
                site_type += len(ways)*[type_ind]
            
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