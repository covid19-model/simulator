import os

'''
Runs several experiments on cluster
'''

if __name__ == '__main__':

    # sim-tracing-isolation
    os.system('python sim-tracing-isolation.py --country GER --area TU')
    os.system('python sim-tracing-isolation.py --country GER --area KL')
    os.system('python sim-tracing-isolation.py --country GER --area RH')
    os.system('python sim-tracing-isolation.py --country GER --area RH')
    os.system('python sim-tracing-isolation.py --country CH --area VD')
    os.system('python sim-tracing-isolation.py --country CH --area BE')
    os.system('python sim-tracing-isolation.py --country CH --area TI')
    os.system('python sim-tracing-isolation.py --country CH --area JU')

    
