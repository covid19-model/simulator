import os, argparse
from concurrent.futures import ProcessPoolExecutor

'''
Runs several experiments on cluster
'''

def sim_narrowcasting(country, area):
    os.system(f'python sim-narrowcasting.py --country {country} --area {area}')

if __name__ == '__main__':

    locs = [('CH', 'TI'), ('GER', 'TU'), ('CH', 'BE'),
            ('GER', 'KL'), ('CH', 'JU'), ('GER', 'RH')]

    countries, areas = zip(*locs)
    with ProcessPoolExecutor(6) as ex:
        res = ex.map(sim_narrowcasting, countries, areas)

