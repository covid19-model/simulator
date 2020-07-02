import os

'''
Runs several experiments on cluster
'''

if __name__ == '__main__':

    locs = [
        ('GER', 'TU'), ('GER', 'KL'), ('GER', 'RH'), ('GER', 'TR'),
        ('CH', 'VD'), ('CH', 'BE'), ('CH', 'TI'), ('CH', 'JU'),
    ]

    # sim-baseline
    # baseline for uncontrolled pandemic after lockdown
    for country, area in locs:
        os.system(f'python sim-baseline.py --country {country} --area {area}')

    # sim-tracing-isolation
    # contact tracing with isolation of individuals both from site and from home
    for country, area in locs:
        os.system(f'python sim-tracing-isolation.py --country {country} --area {area}')

    # sim-tracing-isolation-compliance
    # contact tracing with isolation of individuals both from site and from home, only with partially compliance/adoption
    for country, area in locs:
        os.system(f'python sim-tracing-isolation-compliance.py --country {country} --area {area}')

    # sim-tracing-testing
    # contact tracing with testing of traced individuals
    for country, area in locs:
        os.system(f'python sim-tracing-testing.py --country {country} --area {area}')

    # sim-conditional-measures
    # social distancing measures active only when daily case count goes above a certain threshold of cases per 100k inhabitants
    for country, area in locs:
        os.system(f'python sim-conditional-measures.py --country {country} --area {area}')
    
    # sim-k-groups
    # alternating curfews for K random groups of the population
    for country, area in locs:
        os.system(f'python sim-k-groups.py --country {country} --area {area}')

    # sim-continued-lockdown
    # exploring the effect of different lengths of continued lockdown periods starting at the end of lockdown in reality
    for country, area in locs:
        os.system(f'python sim-continued-lockdown.py --country {country} --area {area}')

    # sim-vulnerable-groups
    # exploring the effect of protection of vulnerable groups of the population only
    for country, area in locs:
        os.system(f'python sim-vulnerable-groups.py --country {country} --area {area}')

    
