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
    locs = [('CH', 'TI'), ('CH', 'BE'), ('CH', 'JU')]

    # 2
    # locs = [('GER', 'TU'), ('GER', 'KL'), ('GER', 'RH')]

    for country, area in locs:

        # sim-tracing-isolation
        # contact tracing with isolation of individuals both from site and from home
        os.system(f'python sim-tracing-isolation.py --country {country} --area {area}')

        # sim-conditional-measures-scenario-b
        # social distancing measures active only when daily case count goes above a certain threshold of cases per 100k inhabitants
        os.system(f'python sim-conditional-measures-scenario-b.py --country {country} --area {area}')

        # sim-tracing-isolation-compliance
        # contact tracing with isolation of individuals both from site and from home, only with partially compliance/adoption
        os.system(f'python sim-tracing-isolation-compliance.py --country {country} --area {area}')

        # sim-tracing-testing
        # contact tracing with testing of traced individuals
        os.system(f'python sim-tracing-testing.py --country {country} --area {area}')

        # sim-baseline-scenario-b
        # baseline for uncontrolled pandemic scenario B
        os.system(f'python sim-baseline-scenario-b.py --country {country} --area {area}')
        
        
