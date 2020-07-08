import os

'''
Runs several experiments on cluster
'''

if __name__ == '__main__':

    # locs = [
    #     ('GER', 'TU'), ('GER', 'KL'), ('GER', 'RH'), ('GER', 'TR'),
    #     ('CH', 'VD'), ('CH', 'BE'), ('CH', 'TI'), ('CH', 'JU'),
    # ]

    # 1
    # locs = [('CH', 'TI'), ('CH', 'BE'), ('CH', 'JU')]

    # 2
    # locs = [('GER', 'TU'), ('GER', 'KL'), ('GER', 'RH')]

    locs = [('CH', 'TI'), ('GER', 'TU'), ('CH', 'BE'),
            ('GER', 'KL'), ('CH', 'JU'), ('GER', 'RH')]

    for country, area in locs:

        # sim-conditional-measures
        # social distancing measures active only when daily case count goes above a certain threshold of cases per 100k inhabitants
        os.system(f'python sim-conditional-measures.py --country {country} --area {area}')

        # sim-continued-lockdown
        # exploring the effect of different lengths of continued lockdown periods starting at the end of lockdown in reality
        os.system(f'python sim-continued-lockdown.py --country {country} --area {area}')

        # sim-k-groups
        # alternating curfews for K random groups of the population
        os.system(f'python sim-k-groups.py --country {country} --area {area}')

        # sim-vulnerable-groups
        # exploring the effect of protection of vulnerable groups of the population only
        os.system(f'python sim-vulnerable-groups.py --country {country} --area {area}')

        # sim-baseline
        # baseline for uncontrolled pandemic after lockdown
        os.system(f'python sim-baseline.py --country {country} --area {area}')
            
        
        
