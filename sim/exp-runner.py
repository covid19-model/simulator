import os, argparse

'''
Runs several experiments on cluster
'''

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--option", required=True, type=int)
    args = parser.parse_args()


    locs = [('CH', 'TI'), ('GER', 'TU'), ('CH', 'BE'),
            ('GER', 'KL'), ('CH', 'JU'), ('GER', 'RH')]

    if args.option == 0:
        for country, area in locs:
            os.system(f'python sim-tracing.py --country {country} --area {area}')

    if args.option == 1:
        for country, area in locs:
            os.system(f'python sim-tracing-compliance.py --country {country} --area {area}')

    if args.option == 2:
        for country, area in locs:
            os.system(f'python sim-conditional-measures-scenario-b.py --country {country} --area {area}')

    if args.option == 3:
        for country, area in locs:
            os.system(f'python sim-baseline-scenario-b.py --country {country} --area {area}')

    if args.option == 4:
        os.system(f'python sim-baseline.py--country CH --area JU')
        os.system(f'python sim-baseline.py--country CH --area BE')
        os.system(f'python sim-baseline.py--country GER --area KL')

        os.system(f'python sim-continued-lockdown.py --country CH --area JU')
        os.system(f'python sim-vulnerable-groups.py --country CH --area JU')
        os.system(f'python sim-k-groups.py --country CH --area JU')



        
        
        
