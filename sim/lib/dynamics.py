
import time
import bisect
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
import scipy as sp
import os, math
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sympy import Symbol, integrate, lambdify, exp, Max, Min, Piecewise, log
import pprint

from lib.priorityqueue import PriorityQueue
from lib.measures import * 

TO_HOURS = 24.0

class DiseaseModel(object):
    """
    Simulate continuous-time SEIR epidemics with exponentially distributed inter-event times.
    All units in the simulator are in hours for numerical stability, though disease parameters are
    assumed to be in units of days as usual in epidemiology
    """

    def __init__(self, mob, distributions):
        """
        Init simulation object with parameters

        Arguments:
        ---------
        mob:
            object of class MobilitySimulator providing mobility data

        """ 

        # cache settings
        self.mob = mob
        self.d = distributions
        assert(np.allclose(np.array(self.d.delta), np.array(self.mob.delta), atol=1e-3))

        # parse distributions object
        self.lambda_0 = self.d.lambda_0
        self.gamma = self.d.gamma
        self.fatality_rates_by_age = self.d.fatality_rates_by_age
        self.p_hospital_by_age = self.d.p_hospital_by_age
        self.delta = self.d.delta

        # parse mobility object
        self.n_people = mob.num_people
        self.n_sites = mob.num_sites
        self.max_time = mob.max_time

        # special state variables from mob object 
        self.people_age = mob.people_age
        self.num_age_groups = mob.num_age_groups
        self.site_type = mob.site_type
        self.site_dict = mob.site_dict
        self.num_site_types = mob.num_site_types
        
        self.people_household = mob.people_household  # j-th entry is household index of individual j
        self.households = mob.households              # {household index: [individuals in household]}
            
        assert(self.num_age_groups == self.fatality_rates_by_age.shape[0])
        assert(self.num_age_groups == self.p_hospital_by_age.shape[0])

        # print
        self.last_print = time.time()
        self._PRINT_INTERVAL = 0.1
        self._PRINT_MSG = (
            't: {t:.2f} '
            '| '
            '{maxt:.2f} hrs '
            '({maxd:.0f} d)'
            )

    def __print(self, t, force=False):
        if ((time.time() - self.last_print > self._PRINT_INTERVAL) or force) and self.verbose:
            print('\r', self._PRINT_MSG.format(t=t, maxt=self.max_time, maxd=self.max_time / 24),
                  sep='', end='', flush=True)
            self.last_print = time.time()
    

    def __init_run(self):
        """
        Initialize the run of the epidemic
        """

        self.queue = PriorityQueue()
        self.testing_queue = PriorityQueue()

        '''
        State and queue codes (transition event into this state)

        'susc': susceptible
        'expo': exposed
        'ipre': infectious pre-symptomatic
        'isym': infectious symptomatic
        'iasy': infectious asymptomatic
        'posi': tested positive
        'nega': tested negative
        'resi': resistant
        'dead': dead
        'hosp': hospitalized

        'test': event of i getting a test (transitions to posi if not susc)
        'execute_tests': generic event indicating that testing queue should be processed

        '''
        self.legal_states = ['susc', 'expo', 'ipre', 'isym', 'iasy', 'posi', 'nega', 'resi', 'dead', 'hosp']
        self.legal_preceeding_state = {
            'expo' : ['susc',],
            'ipre' : ['expo',],
            'isym' : ['ipre',],
            'iasy' : ['expo',],
            'posi' : ['isym', 'ipre', 'iasy', 'expo'],
            'nega' : ['susc', 'resi'],
            'resi' : ['isym', 'iasy'],
            'dead' : ['isym',],
            'hosp' : ['isym',],
        }

        self.state = {
            'susc': np.ones(self.n_people, dtype='bool'),
            'expo': np.zeros(self.n_people, dtype='bool'),
            'ipre': np.zeros(self.n_people, dtype='bool'),
            'isym': np.zeros(self.n_people, dtype='bool'),
            'iasy': np.zeros(self.n_people, dtype='bool'),
            'posi': np.zeros(self.n_people, dtype='bool'),
            'nega': np.zeros(self.n_people, dtype='bool'),
            'resi': np.zeros(self.n_people, dtype='bool'),
            'dead': np.zeros(self.n_people, dtype='bool'),
            'hosp': np.zeros(self.n_people, dtype='bool'),
        }

        self.state_started_at = {
            'susc': - np.inf * np.ones(self.n_people, dtype='float'),
            'expo': np.inf * np.ones(self.n_people, dtype='float'),
            'ipre': np.inf * np.ones(self.n_people, dtype='float'),
            'isym': np.inf * np.ones(self.n_people, dtype='float'),
            'iasy': np.inf * np.ones(self.n_people, dtype='float'),
            'posi': np.inf * np.ones(self.n_people, dtype='float'),
            'nega': np.inf * np.ones(self.n_people, dtype='float'),
            'resi': np.inf * np.ones(self.n_people, dtype='float'),
            'dead': np.inf * np.ones(self.n_people, dtype='float'),
            'hosp': np.inf * np.ones(self.n_people, dtype='float'),
        }
        self.state_ended_at = {
            'susc': np.inf * np.ones(self.n_people, dtype='float'),
            'expo': np.inf * np.ones(self.n_people, dtype='float'),
            'ipre': np.inf * np.ones(self.n_people, dtype='float'),
            'isym': np.inf * np.ones(self.n_people, dtype='float'),
            'iasy': np.inf * np.ones(self.n_people, dtype='float'),
            'posi': np.inf * np.ones(self.n_people, dtype='float'),
            'nega': np.inf * np.ones(self.n_people, dtype='float'),
            'resi': np.inf * np.ones(self.n_people, dtype='float'),
            'dead': np.inf * np.ones(self.n_people, dtype='float'),
            'hosp': np.inf * np.ones(self.n_people, dtype='float'),
        }   
        self.outcome_of_test = np.zeros(self.n_people, dtype='bool')

        # infector of i
        self.parent = -1 * np.ones(self.n_people, dtype='int')

        # no. people i infected (given i was in a certain state)
        self.children_count_iasy = np.zeros(self.n_people, dtype='int')
        self.children_count_ipre = np.zeros(self.n_people, dtype='int')
        self.children_count_isym = np.zeros(self.n_people, dtype='int')
        
        # contact tracing
        # records which contact caused the exposure of `i`
        self.contact_caused_expo = [None for i in range(self.n_people)]
        # list of tuples (i, contacts) where `contacts` were valid when `i` got tested positive
        self.valid_contacts_for_tracing = []
        # evaluates an integral of the exposure rate 
        self.exposure_integral = self.make_exposure_int_eval()
        self.exposure_rate = self.make_exposure_rate_eval() # for sanity check

        # DEBUG
        self.risk_got_exposed = np.zeros(11)
        self.risk_got_not_exposed = np.zeros(11)


    def initialize_states_for_seeds(self):
        """
        Sets state variables according to invariants as given by `self.initial_seeds`

        NOTE: by the seeding heuristic using the reproductive rate
        we assume that exposures already took place
        """
        assert(isinstance(self.initial_seeds, dict))
        for state, seeds_ in self.initial_seeds.items():
            for i in seeds_:
                assert(self.was_initial_seed[i] == False)
                self.was_initial_seed[i] = True
                
                # inital exposed
                if state == 'expo':
                    self.__process_exposure_event(t=0.0, i=i, parent=None, contact=None)

                # initial presymptomatic
                elif state == 'ipre':
                    self.state['susc'][i] = False
                    self.state['expo'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 0

                    # no exposures added due to heuristic `expo` seeds using reproductive rate
                    self.__process_presymptomatic_event(0.0, i, add_exposures=False) 


                # initial asymptomatic
                elif state == 'iasy':

                    self.state['susc'][i] = False
                    self.state['expo'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 1

                    # no exposures added due to heuristic `expo` seeds using reproductive rate
                    self.__process_asymptomatic_event(0.0, i, add_exposures=False)

                # initial symptomatic
                elif state == 'isym' or state == 'isym_notposi':

                    self.state['susc'][i] = False
                    self.state['ipre'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0
                    self.state_ended_at['expo'][i] = -1.0
                    self.state_started_at['ipre'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_symptomatic_event(0.0, i)

                # initial symptomatic and positive
                elif state == 'isym_posi':

                    self.state['susc'][i] = False
                    self.state['ipre'][i] = True
                    self.state['posi'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0
                    self.state_ended_at['expo'][i] = -1.0
                    self.state_started_at['ipre'][i] = -1.0
                    self.state_started_at['posi'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_symptomatic_event(0.0, i, apply_for_test=False)

                # initial resistant and positive
                elif state == 'resi_posi':

                    self.state['susc'][i] = False
                    self.state['isym'][i] = True
                    self.state['posi'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0
                    self.state_ended_at['expo'][i] = -1.0
                    self.state_started_at['ipre'][i] = -1.0
                    self.state_ended_at['ipre'][i] = -1.0
                    self.state_started_at['isym'][i] = -1.0
                    self.state_started_at['posi'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_resistant_event(0.0, i)

                # initial resistant and positive
                elif state == 'resi_notposi':

                    self.state['susc'][i] = False
                    self.state['isym'][i] = True

                    self.state_ended_at['susc'][i] = -1.0
                    self.state_started_at['expo'][i] = -1.0
                    self.state_ended_at['expo'][i] = -1.0
                    self.state_started_at['ipre'][i] = -1.0
                    self.state_ended_at['ipre'][i] = -1.0
                    self.state_started_at['isym'][i] = -1.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_resistant_event(0.0, i)

                else:
                    raise ValueError('Invalid initial seed state.')

    def make_exposure_int_eval(self):
        '''
        Returns evaluatable numpy function that computes an integral
        of the exposure rate. The function returned takes the following arguments

            `j_from`:     visit start of j
            `j_to`:       visit end of j
            `inf_from`:   visit start of infector
            `inf_to`:     visit end of infector
            `beta_site`:  transmission rate at site

        '''

        # define symbols in exposure rate
        beta_sp = Symbol('beta')
        base_rate_sp = Symbol('base_rate')
        lower_sp = Symbol('lower')
        upper_sp = Symbol('upper')
        a_sp = Symbol('a')
        b_sp = Symbol('b')
        u_sp = Symbol('u')
        t_sp = Symbol('t')

        # symbolically integrate term of the exposure rate over [lower_sp, upper_sp]
        expo_int_symb = Max(integrate(
            beta_sp * 
            integrate(
                base_rate_sp *
                self.gamma *
                Piecewise((1.0, (a_sp <= u_sp) & (u_sp <= b_sp)), (0.0, True)) * 
                exp(- self.gamma * (t_sp - u_sp)), 
            (u_sp, t_sp - self.delta, t_sp)),
        (t_sp, lower_sp, upper_sp)
        ).simplify(), 0.0)

        f_sp = lambdify((lower_sp, upper_sp, a_sp, b_sp, beta_sp, base_rate_sp), expo_int_symb, 'numpy')

        # define function with named arguments
        def f(*, j_from, j_to, inf_from, inf_to, beta_site, base_rate):
            '''Shifts to 0.0 for numerical stability'''
            return f_sp(0.0, j_to - j_from, inf_from - j_from, inf_to - j_from, beta_site, base_rate)

        return f
    
    def make_exposure_rate_eval(self):
        '''
        Returns evaluatable numpy function that computes an integral
        of the exposure rate. The function returned takes the following arguments

            `inf_from`:   visit start of infector
            `inf_to`:     visit end of infector
            `beta_site`:  transmission rate at site

        '''

        # define symbols in exposure rate
        a_sp = Symbol('a')
        b_sp = Symbol('b')
        u_sp = Symbol('u')
        t_sp = Symbol('t')

        # symbolically integrate term of the exposure rate over [lower_sp, upper_sp]
        expo_rate_symb = Max(
            integrate(
                self.gamma * \
                Piecewise((1.0, (a_sp <= u_sp) & (u_sp <= b_sp)), (0.0, True)) \
                    * exp(- self.gamma * (t_sp - u_sp)),
                (u_sp, t_sp - self.delta, t_sp)).simplify(),
        0.0)

        f_sp = lambdify((t_sp, a_sp, b_sp), expo_rate_symb, 'numpy')

        # define function with named arguments
        def f(*, t, inf_from, inf_to):
            return f_sp(t, inf_from, inf_to)

        return f

    def launch_epidemic(self, params, initial_counts, testing_params, measure_list, thresholds_roc=[], verbose=True):
        """
        Run the epidemic, starting from initial event list.
        Events are treated in order in a priority queue. An event in the queue is a tuple
        the form
            `(time, event_type, node, infector_node, location, metadata)`

        """
        self.verbose = verbose
        self.thresholds_roc = thresholds_roc

        # optimized params
        self.betas = params['betas']
        self.mu = self.d.mu
        self.alpha = self.d.alpha
        
        # household param
        if 'beta_household' in params:
            self.beta_household = params['beta_household']
        else:
            self.beta_household = 0.0

        # testing settings
        self.testing_frequency  = testing_params['testing_frequency']
        self.test_targets       = testing_params['test_targets']
        self.test_queue_policy  = testing_params['test_queue_policy']
        self.test_reporting_lag = testing_params['test_reporting_lag']        
        self.tests_per_batch    = testing_params['tests_per_batch']
        self.testing_t_window   = testing_params['testing_t_window']
        self.t_pos_tests = []
        self.test_fpr = testing_params['test_fpr']
        self.test_fnr = testing_params['test_fnr']
        
        # smart tracing settings
        self.smart_tracing_actions             = testing_params['smart_tracing_actions']
        self.smart_tracing_contact_delta       = testing_params['smart_tracing_contact_delta']

        self.smart_tracing_policy_isolate      = testing_params['smart_tracing_policy_isolate']
        self.smart_tracing_isolation_duration  = testing_params['smart_tracing_isolation_duration']
        self.smart_tracing_isolated_contacts   = testing_params['smart_tracing_isolated_contacts']
        self.smart_tracing_isolation_threshold = testing_params['smart_tracing_isolation_threshold']

        self.smart_tracing_policy_test         = testing_params['smart_tracing_policy_test']
        self.smart_tracing_tested_contacts     = testing_params['smart_tracing_tested_contacts']
        self.smart_tracing_testing_threshold   = testing_params['smart_tracing_testing_threshold']
        self.trigger_tracing_after_posi_trace_test = testing_params['trigger_tracing_after_posi_trace_test']

        self.smart_tracing_stats_window = testing_params.get(
            'smart_tracing_stats_window', (0.0, self.max_time))

        self.smart_tracing_p_willing_to_share  = testing_params['p_willing_to_share']

        if 'isolate' in self.smart_tracing_actions \
            and self.smart_tracing_isolated_contacts == 0:
            print('Warning: `smart_tracing_isolated_contacts` is 0 even though '
                    'the policy ought to isolate traced contacts')
        
        if 'test' in self.smart_tracing_actions \
            and self.smart_tracing_tested_contacts == 0:
            print('Warning: `smart_tracing_isolated_contacts` is 0 even though '
                    'the policy ought to isolate traced contacts')

        if self.smart_tracing_policy_isolate == 'advanced-threshold' \
            and self.smart_tracing_isolation_threshold == 1.0:
            print('Warning: `smart_tracing_isolation_threshold` is 1.0 even though '
                    'the policy ought to isolate contacts with empirical risk above the threshold')
        
        if self.smart_tracing_policy_test == 'advanced-threshold' \
            and self.smart_tracing_testing_threshold == 1.0:
            print('Warning: `smart_tracing_testing_threshold` is 1.0 even though '
                    'the policy ought to test contacts with empirical risk above the threshold')

        
        # Set list of measures
        if not isinstance(measure_list, MeasureList):
            raise ValueError("`measure_list` must be a `MeasureList` object")
        self.measure_list = measure_list

        # Sample bernoulli outcome for all SocialDistancingForAllMeasure and conditional measures
        self.measure_list.init_run(SocialDistancingForAllMeasure,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))

        self.measure_list.init_run(SocialDistancingBySiteTypeForAllMeasure,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))

        self.measure_list.init_run(UpperBoundCasesSocialDistancing,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))

        self.measure_list.init_run(UpperBoundCasesBetaMultiplier,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))

        self.measure_list.init_run(SocialDistancingPerStateMeasure,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))
        
        self.measure_list.init_run(SocialDistancingForPositiveMeasure,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))
        
        self.measure_list.init_run(SocialDistancingForPositiveMeasureHousehold)

        self.measure_list.init_run(SocialDistancingByAgeMeasure,
                                   num_age_groups=self.num_age_groups,
                                   n_visits=max(self.mob.visit_counts))
        
        self.measure_list.init_run(ComplianceForAllMeasure,
                                   n_people=self.n_people)

        self.measure_list.init_run(ManualTracingReachabilityForAllMeasure,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))

        self.measure_list.init_run(ManualTracingForAllMeasure,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))
        
        self.measure_list.init_run(SocialDistancingForSmartTracing,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))    

        self.measure_list.init_run(SocialDistancingForSmartTracingHousehold,
                                   n_people=self.n_people)
        
        self.measure_list.init_run(SocialDistancingSymptomaticAfterSmartTracing,
                                   n_people=self.n_people)

        self.measure_list.init_run(SocialDistancingSymptomaticAfterSmartTracingHousehold,
                                   n_people=self.n_people)

        self.measure_list.init_run(SocialDistancingForKGroups)

        # if specified, scale optimized betas a priori
        try:
            apriori_beta = self.measure_list.find_first(APrioriBetaMultiplierMeasureByType)
            for k in range(self.num_site_types):
                self.betas[self.site_dict[k]] *= apriori_beta.beta_factor(typ=self.site_dict[k])
        except:
            pass

        # for analysis purposes, compute mean of betas weighted by number of times each site type occurs
        # i.e. mean(rel_occurrence_of_site_type[k] * beta[k])
        self.betas_weighted_mean = sum([
            self.betas[self.site_dict[k]] 
            * np.sum(self.site_type == k) / self.n_sites # relative frequency of site type k
            for k in range(self.num_site_types)
        ]) 

        # init state variables with seeds
        self.__init_run()
        self.was_initial_seed = np.zeros(self.n_people, dtype='bool')

        total_seeds = sum(v for v in initial_counts.values())
        initial_people = np.random.choice(self.n_people, size=total_seeds, replace=False)
        ptr = 0
        self.initial_seeds = dict()
        for k, v in initial_counts.items():
            self.initial_seeds[k] = initial_people[ptr:ptr + v].tolist()
            ptr += v          

        # sample all iid events ahead of time in batch
        batch_size = (self.n_people, )
        self.delta_expo_to_ipre = self.d.sample_expo_ipre(size=batch_size)
        self.delta_ipre_to_isym = self.d.sample_ipre_isym(size=batch_size)
        self.delta_isym_to_resi = self.d.sample_isym_resi(size=batch_size)
        self.delta_isym_to_dead = self.d.sample_isym_dead(size=batch_size)
        self.delta_expo_to_iasy = self.d.sample_expo_iasy(size=batch_size)
        self.delta_iasy_to_resi = self.d.sample_iasy_resi(size=batch_size)
        self.delta_isym_to_hosp = self.d.sample_isym_hosp(size=batch_size)

        self.bernoulli_is_iasy = np.random.binomial(1, self.alpha, size=batch_size)
        self.bernoulli_is_fatal = self.d.sample_is_fatal(self.people_age, size=batch_size)
        self.bernoulli_is_hospi = self.d.sample_is_hospitalized(self.people_age, size=batch_size)
            

        # initial seed
        self.initialize_states_for_seeds()

        # not initially seeded
        if self.lambda_0 > 0.0:
            # sample non-contact exposure events
            delta_susc_to_expo = self.d.sample_susc_baseexpo(size=self.n_people)
            for i in range(self.n_people):
                if not self.was_initial_seed[i] and delta_susc_to_expo[i] < self.max_time: 
                    self.queue.push(
                        (delta_susc_to_expo[i], 'expo', i, None, None, None),
                        priority=delta_susc_to_expo[i])

        # initialize test processing events: add 'update_test' event to queue for `testing_frequency` hour
        for h in range(1, math.floor(self.max_time / self.testing_frequency)):
            ht = h * self.testing_frequency
            self.queue.push((ht, 'execute_tests', None, None, None, None), priority=ht)

        # MAIN EVENT LOOP
        t = 0.0
        while self.queue:

            # get next event to process
            t, event, i, infector, k, metadata = self.queue.pop()

            # check if testing processing
            if event == 'execute_tests':
                self.__update_testing_queue(t)
                continue

            # check termination
            if t > self.max_time:
                t = self.max_time
                self.__print(t, force=True)
                if self.verbose:
                    print(f'\n[Reached max time: {int(self.max_time)}h ({int(self.max_time // 24)}d)]')
                break
            if np.sum((1 - self.state['susc']) * (self.state['resi'] + self.state['dead'])) == self.n_people:
                if self.verbose:
                    print('\n[Simulation ended]')
                break

            # process event
            if event == 'expo':
                i_susceptible = ((not self.state['expo'][i]) and (self.state['susc'][i]))

                # base rate exposure
                if (infector is None) and i_susceptible:
                    self.__process_exposure_event(t=t, i=i, parent=None, contact=None)

                # household exposure
                if (infector is not None) and i_susceptible and k == -1:
                    
                    # 1) check whether infector recovered or dead
                    infector_recovered = \
                        (self.state['resi'][infector] or 
                            self.state['dead'][infector])

                    # 2) check whether infector got hospitalized
                    infector_hospitalized = self.state['hosp'][infector]

                    # 3) check whether infector or i are not at home
                    infector_away_from_home = False
                    i_away_from_home = False

                    infector_visits = self.mob.mob_traces[infector].find((t, t))
                    i_visits = self.mob.mob_traces[i].find((t, t))

                    for v in infector_visits:
                        infector_away_from_home = \
                            ((v.t_to > t) and # infector actually at a site, not just matching "environmental contact"
                             (not self.is_person_home_from_visit_due_to_measure(
                             t=t, i=infector, visit_id=v.id, site_type=self.site_dict[self.site_type[v.site]])))
                        if infector_away_from_home:
                            break

                    for v in i_visits:
                        i_away_from_home = i_away_from_home or \
                            ((v.t_to > t) and # i actually at a site, not just matching "environmental contact"
                             (not self.is_person_home_from_visit_due_to_measure(
                             t=t, i=i, visit_id=v.id, site_type=self.site_dict[self.site_type[v.site]])))

                    away_from_home = (infector_away_from_home or i_away_from_home)
                    
                    # 4) check whether infector or i were isolated from household members
                    infector_isolated_at_home = (
                        self.measure_list.is_contained(
                            SocialDistancingForPositiveMeasureHousehold, t=t,
                            j=infector, 
                            state_posi_started_at=self.state_started_at['posi'], 
                            state_posi_ended_at=self.state_ended_at['posi'], 
                            state_resi_started_at=self.state_started_at['resi'], 
                            state_dead_started_at=self.state_started_at['dead']) or
                        self.measure_list.is_contained(
                            SocialDistancingForSmartTracingHousehold, t=t,
                            state_nega_started_at=self.state_started_at['nega'],
                            state_nega_ended_at=self.state_ended_at['nega'],
                            j=infector) or 
                        self.measure_list.is_contained(
                            SocialDistancingSymptomaticAfterSmartTracingHousehold, t=t,
                            state_isym_started_at=self.state_started_at['isym'],
                            state_isym_ended_at=self.state_ended_at['isym'],
                            state_nega_started_at=self.state_started_at['nega'],
                            state_nega_ended_at=self.state_ended_at['nega'],
                            j=infector)
                    )

                    i_isolated_at_home = (
                        self.measure_list.is_contained(
                            SocialDistancingForPositiveMeasureHousehold, t=t,
                            j=i,
                            state_posi_started_at=self.state_started_at['posi'],
                            state_posi_ended_at=self.state_ended_at['posi'],
                            state_resi_started_at=self.state_started_at['resi'],
                            state_dead_started_at=self.state_started_at['dead']) or
                        self.measure_list.is_contained(
                            SocialDistancingForSmartTracingHousehold, t=t,
                            state_nega_started_at=self.state_started_at['nega'],
                            state_nega_ended_at=self.state_ended_at['nega'],
                            j=i) or
                        self.measure_list.is_contained(
                            SocialDistancingSymptomaticAfterSmartTracingHousehold, t=t,
                            state_isym_started_at=self.state_started_at['isym'],
                            state_isym_ended_at=self.state_ended_at['isym'],
                            state_nega_started_at=self.state_started_at['nega'],
                            state_nega_ended_at=self.state_ended_at['nega'],
                            j=i)
                    )

                    somebody_isolated = (infector_isolated_at_home or i_isolated_at_home)

                    # "thinning"
                    # if none of 1), 2), 3), 4) are true, the event is valid
                    if  (not infector_recovered) and \
                        (not infector_hospitalized) and \
                        (not away_from_home) and \
                        (not somebody_isolated):

                        self.__process_exposure_event(t=t, i=i, parent=infector, contact=None)

                    # if 2), 3), or 4) were true, i.e. infector not recovered,
                    # a household exposure could happen at a later point, hence sample a new event
                    if (infector_hospitalized or away_from_home or somebody_isolated):

                        # find tmax for efficiency
                        if self.state['iasy'][infector]:
                            base_rate_infector = self.mu
                            tmax = self.state_started_at['iasy'][infector] + self.delta_iasy_to_resi[infector]
                        else:
                            base_rate_infector = 1.0
                            tmax = (self.state_started_at['ipre'][infector] + self.delta_ipre_to_isym[infector] +
                                self.delta_isym_to_dead[infector] if self.bernoulli_is_fatal[infector] else self.delta_isym_to_resi[infector])

                        # sample exposure at later point 
                        if t < tmax:
                            self.__push_household_exposure_infector_to_j(
                                t=t, infector=infector, j=i, base_rate=base_rate_infector, tmax=tmax)

                # contact exposure
                if (infector is not None) and i_susceptible and k >= 0:
                    
                    # for contact exposures, `contact` causing the exposure is passed
                    contact = metadata 
                    i_visit_id, infector_visit_id = contact.id_tup
                    assert(k == contact.site)

                    # is_in_contact2, contact2 = self.mob.is_in_contact(indiv_i=i, indiv_j=infector, site=k, t=t)
                    # assert(is_in_contact2 and (contact == contact2))

                    # 1) check whether infector recovered or dead
                    infector_recovered = \
                        (self.state['resi'][infector] or 
                            self.state['dead'][infector])

                    # 2) check whether infector stayed at home due to measures
                    #    or got hospitalized
                    infector_contained = self.is_person_home_from_visit_due_to_measure(
                        t=t, i=infector, visit_id=infector_visit_id, 
                        site_type=self.site_dict[self.site_type[k]]) \
                    or self.state['hosp'][infector]
                                            
                    # 3) check whether susceptible stayed at home due to measures
                    i_contained = self.is_person_home_from_visit_due_to_measure(
                        t=t, i=i, visit_id=i_visit_id, 
                        site_type=self.site_dict[self.site_type[k]])

                    # 4) check whether infectiousness got reduced due to site specific 
                    #    measures and as a consequence this event didn't occur
                    rejection_prob = self.reject_exposure_due_to_measure(t=t, k=k)
                    site_avoided_infection = (np.random.uniform() < rejection_prob)

                    # "thinning"
                    # if none of 1), 2), 3), 4) are true, the event is valid
                    if (not infector_recovered) and \
                        (not infector_contained) and \
                        (not i_contained) and \
                        (not site_avoided_infection):

                        self.__process_exposure_event(t=t, i=i, parent=infector, contact=contact)

                    # if any of 2), 3), 4) were true, i.e. infector not recovered,
                    # an exposure could happen at a later point, hence sample a new event 
                    if (infector_contained or i_contained or site_avoided_infection):

                        # find tmax for efficiency
                        if self.state['iasy'][infector]:
                            base_rate_infector = self.mu
                            tmax = self.state_started_at['iasy'][infector] + \
                                self.delta_iasy_to_resi[infector]
                        else:
                            base_rate_infector = 1.0
                            tmax = (self.state_started_at['ipre'][infector] + self.delta_ipre_to_isym[infector] +
                                    self.delta_isym_to_dead[infector] if self.bernoulli_is_fatal[infector] else self.delta_isym_to_resi[infector])

                        # sample exposure at later point
                        if t < tmax:
                            self.__push_contact_exposure_infector_to_j(
                                t=t, infector=infector, j=i, base_rate=base_rate_infector, tmax=tmax)                 

            elif event == 'ipre':
                self.__process_presymptomatic_event(t, i)

            elif event == 'iasy':
                self.__process_asymptomatic_event(t, i)

            elif event == 'isym':
                self.__process_symptomatic_event(t, i)

            elif event == 'resi':
                self.__process_resistant_event(t, i)

            elif event == 'test':
                self.__process_testing_event(t, i, metadata)

            elif event == 'dead':
                self.__process_fatal_event(t, i)

            elif event == 'hosp':
                # cannot get hospitalization if not ill anymore 
                valid_hospitalization = \
                    ((not self.state['resi'][i]) and 
                        (not self.state['dead'][i]))

                if valid_hospitalization:
                    self.__process_hosp_event(t, i)
            else:
                # this should only happen for invalid exposure events
                assert(event == 'expo')

            # print
            self.__print(t, force=True)



        # print('% exposed in risk buckets: ', 100.0 * self.risk_got_exposed / (self.risk_got_exposed + self.risk_got_not_exposed))
   
        '''Compute ROC statistics'''
        # tracing_stats [threshold][policy][action][stat]
        self.tracing_stats = {}
        if len(self.thresholds_roc) > 0:
            for threshold in self.thresholds_roc:
                self.tracing_stats[threshold] = self.compute_roc_stats(
                    threshold_isolate=threshold, threshold_test=threshold)

            # stats = self.tracing_stats[self.thresholds_roc[0]]['sites']['isolate']
            
            # print(" P {:5.2f}  N {:5.2f}".format(
            #     (stats['fn'] + stats['tp']), (stats['fp'] + stats['tn'])
            # ))

        # free memory
        self.valid_contacts_for_tracing = None
        self.queue = None


    def compute_roc_stats(self, *, threshold_isolate, threshold_test):
        '''        
        Recovers contacts for which trace/no-trace decision was made.
        Then re-computes TP/FP/TN/FN for different decision thresholds, given 
        the label that a given person with contact got exposed.
        Assumes `advanced-threshold` policy for both isolation and testing.
        '''

        stats = {
            'sites' : {
                'isolate' : {'tp' : 0, 'fp' : 0, 'tn' : 0, 'fn' : 0},
                'test' :    {'tp' : 0, 'fp' : 0, 'tn' : 0, 'fn' : 0},
            },
            'no_sites' : {
                'isolate' : {'tp' : 0, 'fp' : 0, 'tn' : 0, 'fn' : 0},
                'test' :    {'tp' : 0, 'fp' : 0, 'tn' : 0, 'fn' : 0},
            },
        } 
        
        # c[sites/no_sites][isolate/test][False/True][j]
        # i-j contacts due to which j was traced/not traced
        c = {
            'sites' : {
                'isolate': {
                    False: [[] for _ in range(self.n_people)], 
                    True:  [[] for _ in range(self.n_people)],
                },
                'test': {
                    False: [[] for _ in range(self.n_people)],
                    True:  [[] for _ in range(self.n_people)],
                },
            },
            'no_sites' : {
                'isolate': {
                    False: [[] for _ in range(self.n_people)],
                    True:  [[] for _ in range(self.n_people)],
                },
                'test': {
                    False: [[] for _ in range(self.n_people)],
                    True:  [[] for _ in range(self.n_people)],
                },
            },
        }
        
        individuals_traced = set()

        # for each tracing call due to an `infector`, re-compute classification decision (tracing or not)  
        # under the decision threshold `thres`, and record decision in `contacts_caused_tracing_...` arrays by individual
        for t, infector, valid_contacts_with_j in self.valid_contacts_for_tracing:

            # inspect whether the infector was symptomatic or asymptomatic
            if self.state_started_at['iasy'][infector] < np.inf:
                base_rate_inf = self.mu
            else:
                base_rate_inf = 1.0

            # compute empirical survival probability
            emp_survival_prob = {
                'sites' : dict(),
                'no_sites' : dict()
            }
            for j, contacts_j in valid_contacts_with_j.items():
                individuals_traced.add(j)
                emp_survival_prob['sites'][j] = self.__compute_empirical_survival_probability(
                    t=t, i=infector, j=j, contacts_i_j=contacts_j, base_rate=base_rate_inf, ignore_sites=False)
                emp_survival_prob['no_sites'][j] = self.__compute_empirical_survival_probability(
                    t=t, i=infector, j=j, contacts_i_j=contacts_j, base_rate=base_rate_inf, ignore_sites=True)

            # compute tracing decision
            for policy in ['sites', 'no_sites']:
                for action in ['isolate', 'test']:
                    contacts_action, contacts_no_action = self.__tracing_policy_advanced_threshold(
                        t=t, contacts_with_j=valid_contacts_with_j, 
                        threshold=threshold_isolate if action == 'isolate' else threshold_test,
                        emp_survival_prob=emp_survival_prob[policy])

                    for j, contacts_j in contacts_action:
                        c[policy][action][True][j].append((t, set(contacts_j)))
                    for j, contacts_j in contacts_no_action:
                        c[policy][action][False][j].append((t, set(contacts_j)))

        # for each individual considered in tracing, compare label (contact exposure?) with classification (traced due to this contact?)
        for j in individuals_traced:

            j_was_exposed = self.state_started_at['expo'][j] < np.inf
            c_expo = self.contact_caused_expo[j]         

            # skip if `j` got exposed by another source, even though traced (household or background)
            if (c_expo is None) and j_was_exposed:
                continue

            for policy in ['sites', 'no_sites']:
                for action in ['isolate', 'test']:

                    # each time `j` is traced after a contact
                    for timing, c_traced in c[policy][action][True][j]:
                        
                        # ignore FP if there is no way of knowing
                        if self.state_started_at['expo'][j] < timing - self.smart_tracing_contact_delta:
                            continue

                        # and this contact ultimately caused the exposure of `j`
                        # TP
                        # if (c_expo is not None) and (c_expo in c_traced):
                        if self.state_started_at['expo'][j] <= timing \
                            and timing - self.smart_tracing_contact_delta <= self.state_started_at['expo'][j] \
                            and (c_expo is not None):
                            stats[policy][action]['tp'] += 1

                        # otherwise: `j` either wasn't exposed or exposed but by another contact 
                        # FP
                        else:
                            stats[policy][action]['fp'] += 1

                    # each time `j` is not traced after a contact
                    for timing, c_not_traced in c[policy][action][False][j]:

                        # ignore TN if there is no way of knowing
                        if self.state_started_at['expo'][j] < timing - self.smart_tracing_contact_delta:
                            continue

                        # and this contact ultimately caused the exposure of `j`
                        # FN
                        # if (c_expo is not None) and (c_expo in c_not_traced):
                        if self.state_started_at['expo'][j] <= timing \
                            and timing - self.smart_tracing_contact_delta <= self.state_started_at['expo'][j] \
                            and (c_expo is not None):
                            stats[policy][action]['fn'] += 1

                        # otherwise: `j` either wasn't exposed or not exposed but by another contact
                        # TN
                        else:
                            stats[policy][action]['tn'] += 1

        return stats

        

    def __process_exposure_event(self, *, t, i, parent, contact):
        """
        Mark person `i` as exposed at time `t`
        Push asymptomatic or presymptomatic queue event
        """

        # track flags
        assert(self.state['susc'][i])
        self.state['susc'][i] = False
        self.state['expo'][i] = True
        self.state_ended_at['susc'][i] = t
        self.state_started_at['expo'][i] = t
        if parent is not None:
            self.parent[i] = parent
            if self.state['iasy'][parent]:
                self.children_count_iasy[parent] += 1
            elif self.state['ipre'][parent]:
                self.children_count_ipre[parent] += 1
            elif self.state['isym'][parent]:
                self.children_count_isym[parent] += 1
            else:
                assert False, 'only infectous parents can expose person i'


        # decide whether asymptomatic or (pre-)symptomatic
        if self.bernoulli_is_iasy[i]:
            if t + self.delta_expo_to_iasy[i] < self.max_time:
                self.queue.push(
                    (t + self.delta_expo_to_iasy[i], 'iasy', i, None, None, None),
                    priority=t + self.delta_expo_to_iasy[i])
        else:
            if t + self.delta_expo_to_ipre[i] < self.max_time:
                self.queue.push(
                    (t + self.delta_expo_to_ipre[i], 'ipre', i, None, None, None),
                    priority=t + self.delta_expo_to_ipre[i])

        # record which contact caused this exposure event (to check if it was traced for TP/FP/TN/FN computation)
        if contact is not None:
            assert(self.contact_caused_expo[i] is None)
            self.contact_caused_expo[i] = contact

    def __process_presymptomatic_event(self, t, i, add_exposures=True):
        """
        Mark person `i` as presymptomatic at time `t`
        Push symptomatic queue event
        """

        # track flags
        assert(self.state['expo'][i])
        self.state['ipre'][i] = True
        self.state['expo'][i] = False
        self.state_ended_at['expo'][i] = t
        self.state_started_at['ipre'][i] = t

        # symptomatic event
        if t + self.delta_ipre_to_isym[i] < self.max_time:
            self.queue.push(
                (t + self.delta_ipre_to_isym[i], 'isym', i, None, None, None),
                priority=t + self.delta_ipre_to_isym[i])

        if add_exposures:

            # find tmax for efficiency reasons (based on when individual i will not be infectious anymore)
            tmax = (t + self.delta_ipre_to_isym[i] + 
                self.delta_isym_to_dead[i] if self.bernoulli_is_fatal[i] else 
                self.delta_isym_to_resi[i])

            # contact exposure of others
            self.__push_contact_exposure_events(t=t, infector=i, base_rate=1.0, tmax=tmax)
            
            # household exposures
            if self.households is not None and self.beta_household > 0:
                self.__push_household_exposure_events(t=t, infector=i, base_rate=1.0, tmax=tmax)

    def __process_symptomatic_event(self, t, i, apply_for_test=True):
        """
        Mark person `i` as symptomatic at time `t`
        Push resistant queue event
        """

        # track flags
        assert(self.state['ipre'][i])
        self.state['isym'][i] = True
        self.state['ipre'][i] = False
        self.state_ended_at['ipre'][i] = t
        self.state_started_at['isym'][i] = t

        # testing
        if self.test_targets == 'isym' and apply_for_test:
            self.__apply_for_testing(t=t, i=i, priority= -self.max_time + t, trigger_tracing_if_positive=True)

        # hospitalized?
        if self.bernoulli_is_hospi[i]:
            if t + self.delta_isym_to_hosp[i] < self.max_time:
                self.queue.push(
                    (t + self.delta_isym_to_hosp[i], 'hosp', i, None, None, None),
                    priority=t + self.delta_isym_to_hosp[i])

        # resistant event vs fatality event
        if self.bernoulli_is_fatal[i]:
            if t + self.delta_isym_to_dead[i] < self.max_time:
                self.queue.push(
                    (t + self.delta_isym_to_dead[i], 'dead', i, None, None, None),
                    priority=t + self.delta_isym_to_dead[i])
        else:
            if t + self.delta_isym_to_resi[i] < self.max_time:
                self.queue.push(
                    (t + self.delta_isym_to_resi[i], 'resi', i, None, None, None),
                    priority=t + self.delta_isym_to_resi[i])

    def __process_asymptomatic_event(self, t, i, add_exposures=True):
        """
        Mark person `i` as asymptomatic at time `t`
        Push resistant queue event
        """

        # track flags
        assert(self.state['expo'][i])
        self.state['iasy'][i] = True
        self.state['expo'][i] = False
        self.state_ended_at['expo'][i] = t
        self.state_started_at['iasy'][i] = t

        # resistant event
        if t + self.delta_iasy_to_resi[i] < self.max_time:
            self.queue.push(
                (t + self.delta_iasy_to_resi[i], 'resi', i, None, None, None),
                priority=t + self.delta_iasy_to_resi[i])

        if add_exposures:
            # contact exposure of others
            self.__push_contact_exposure_events(t=t, infector=i, base_rate=self.mu, tmax=t + self.delta_iasy_to_resi[i])
            
            # household exposures
            if self.households is not None and self.beta_household > 0:
                self.__push_household_exposure_events(t=t, infector=i, base_rate=self.mu, tmax=t + self.delta_iasy_to_resi[i])

    def __process_resistant_event(self, t, i):
        """
        Mark person `i` as resistant at time `t`
        """

        # track flags
        assert(self.state['iasy'][i] != self.state['isym'][i]) # XOR
        self.state['resi'][i] = True
        self.state_started_at['resi'][i] = t
        
        # infection type
        if self.state['iasy'][i]:
            self.state['iasy'][i] = False
            self.state_ended_at['iasy'][i] = t

        elif self.state['isym'][i]:
            self.state['isym'][i] = False
            self.state_ended_at['isym'][i] = t
        else:
            assert False, 'Resistant only possible after asymptomatic or symptomatic.'

        # hospitalization ends
        if self.state['hosp'][i]:
            self.state['hosp'][i] = False
            self.state_ended_at['hosp'][i] = t

    def __process_fatal_event(self, t, i):
        """
        Mark person `i` as fatality at time `t`
        """

        # track flags
        assert(self.state['isym'][i])
        self.state['dead'][i] = True
        self.state_started_at['dead'][i] = t

        self.state['isym'][i] = False
        self.state_ended_at['isym'][i] = t

        # hospitalization ends
        if self.state['hosp'][i]:
            self.state['hosp'][i] = False
            self.state_ended_at['hosp'][i] = t
    
    def __process_hosp_event(self, t, i):
        """
        Mark person `i` as hospitalized at time `t`
        """

        # track flags
        assert(self.state['isym'][i])
        self.state['hosp'][i] = True
        self.state_started_at['hosp'][i] = t
    

    def __kernel_term(self, a, b, T):
        '''Computes
        \int_a^b gamma * exp(self.gamma * (u - T)) du
        =  exp(- self.gamma * T) (exp(self.gamma * b) - exp(self.gamma * a))
        '''
        return (np.exp(self.gamma * (b - T)) - np.exp(self.gamma * (a - T))) 


    def __push_contact_exposure_events(self, *, t, infector, base_rate, tmax):
        """
        Pushes all exposure events that person `i` causes
        for other people via contacts, using `base_rate` as basic infectivity
        of person `i` (equivalent to `\mu` in model definition)
        """

        # compute all delta-contacts of `infector` with any other individual
        infectors_contacts = self.mob.find_contacts_of_indiv(indiv=infector, tmin=t, tmax=tmax)

        # iterate over contacts and store contact of with each individual `indiv_i` that is still susceptible 
        valid_contacts = set()
        for contact in infectors_contacts:
            if self.state['susc'][contact.indiv_i]:
                if contact not in self.mob.contacts[contact.indiv_i][infector]:
                    self.mob.contacts[contact.indiv_i][infector].update([contact])
                valid_contacts.add(contact.indiv_i)

        # generate potential exposure event for `j` from contact with `infector`
        for j in valid_contacts:
            self.__push_contact_exposure_infector_to_j(t=t, infector=infector, j=j, base_rate=base_rate, tmax=tmax)


    def __push_contact_exposure_infector_to_j(self, *, t, infector, j, base_rate, tmax):
        """
        Pushes the next exposure event that person `infector` causes for person `j`
        using `base_rate` as basic infectivity of person `i` 
        (equivalent to `\mu` in model definition)
        """
        tau = t
        sampled_event = False
        Z = self.__kernel_term(- self.delta, 0.0, 0.0)

        # sample next arrival from non-homogeneous point process
        while self.mob.will_be_in_contact(indiv_i=j, indiv_j=infector, t=tau, site=None) and not sampled_event and tau < min(tmax, self.max_time):
            
            # check if j could get infected from infector at current `tau`
            # i.e. there is `delta`-contact from infector to j (i.e. non-zero intensity)
            has_infectious_contact, _ = self.mob.is_in_contact(indiv_i=j, indiv_j=infector, t=tau, site=None)

            # if yes: do nothing
            if has_infectious_contact:
                pass 

            # if no:       
            else: 
                # directly jump to next contact start of a `delta`-contact (memoryless property)
                next_contact = self.mob.next_contact(indiv_i=j, indiv_j=infector, t=tau, site=None)

                assert(next_contact is not None) # (while loop invariant)
                tau = next_contact.t_from

            # sample event with maximum possible rate (in hours)
            lambda_max = max(self.betas.values()) * base_rate * Z
            lambda_max = max(lambda_max, 1e-8) # lambda_max = 0 is invalid
            tau += np.random.exponential(scale=1.0 / lambda_max)

            # thinning step: compute current lambda(tau) and do rejection sampling
            sampled_at_infectious_contact, sampled_at_contact = self.mob.is_in_contact(indiv_i=j, indiv_j=infector, t=tau, site=None)

            # 1) reject w.p. 1 if there is no more infectious contact at the new time (lambda(tau) = 0)
            if not sampled_at_infectious_contact:
                continue
            
            # 2) compute infectiousness integral in lambda(tau)
            # a. query times that infector was in [tau - delta, tau] at current site `site`
            site = sampled_at_contact.site
            infector_present = self.mob.list_intervals_in_window_individual_at_site(
                indiv=infector, site=site, t0=tau - self.delta, t1=tau)

            # b. compute contributions of infector being present in [tau - delta, tau]
            intersections = [(max(tau - self.delta, interv.left), min(tau, interv.right))
                for interv in infector_present]
            beta_k = self.betas[self.site_dict[self.site_type[site]]]
            p = (beta_k * base_rate * sum([self.__kernel_term(v[0], v[1], tau) for v in intersections])) \
                / lambda_max

            assert(p <= 1 + 1e-8 and p >= 0)

            # accept w.prob. lambda(t) / lambda_max
            u = np.random.uniform()
            if u <= p and tau < min(tmax, self.max_time):
                self.queue.push(
                    (tau, 'expo', j, infector, site, 
                    sampled_at_contact), # meta info: contact causing infection
                    priority=tau)
                sampled_event = True
        
        # DEBUG
        # all_contacts = []
        # tdebug = t
        # while self.mob.will_be_in_contact(indiv_i=j, indiv_j=infector, t=tdebug, site=None) and tdebug < min(tmax, self.max_time):
        #     c = self.mob.next_contact(indiv_i=j, indiv_j=infector, t=tdebug, site=None)
        #     all_contacts.append(c)
        #     tdebug = c.t_to + 1e-3

        # survival = self.__compute_empirical_survival_probability(
        #     t=tdebug, i=infector, j=j, contacts_i_j=all_contacts, base_rate=base_rate, ignore_sites=False)

        # risk = 1 - survival
        # bucket = np.digitize(risk, np.linspace(0.0, 1.0, num=10, endpoint=False))
        # if sampled_event:
        #     self.risk_got_exposed[bucket] += 1
        # else:
        #     self.risk_got_not_exposed[bucket] += 1
            
            
    def __push_household_exposure_events(self, *, t, infector, base_rate, tmax):
        """
        Pushes all exposure events that person `i` causes
        in the household, using `base_rate` as basic infectivity
        of person `i` (equivalent to `\mu` in model definition)
        """

        def valid_j():
            '''Generates indices j where `infector` is present
            at least `self.delta` hours before j '''
            for j in self.households[self.people_household[infector]]:
                if self.state['susc'][j]:
                    yield j

        # generate potential exposure event for `j` from contact with `infector`
        for j in valid_j():
            self.__push_household_exposure_infector_to_j(t=t, infector=infector, j=j, base_rate=base_rate, tmax=tmax)

    def __push_household_exposure_infector_to_j(self, *, t, infector, j, base_rate, tmax):
        """
        Pushes the next exposure event that person `infector` causes for person `j`,
        who lives in the same household, using `base_rate` as basic infectivity of 
        person `i` (equivalent to `\mu` in model definition)

        We ignore the kernel for households infections since households members
        will overlap for long periods of time at home
        """

        lambda_household = self.beta_household * base_rate * self.__kernel_term(- self.delta, 0.0, 0.0)
        tau = t + np.random.exponential(scale=1.0 / lambda_household)

        # site = -1 means it is a household infection
        # thinning is done at exposure time if needed
        if tau < min(tmax, self.max_time):
            self.queue.push((tau, 'expo', j, infector, -1, None), priority=tau)


    def reject_exposure_due_to_measure(self, t, k):
        '''
        Returns rejection probability of exposure event not occuring
        at location k at time k
        Searches through BetaMultiplierMeasures and retrieves beta multipliers
        Scaling beta is equivalent to scaling down the acceptance probability
        '''

        acceptance_prob = 1.0

        # BetaMultiplierMeasures
        beta_mult_measure = self.measure_list.find(BetaMultiplierMeasureBySite, t=t)
        acceptance_prob *= beta_mult_measure.beta_factor(k=k, t=t) \
            if beta_mult_measure else 1.0

        beta_mult_measure = self.measure_list.find(BetaMultiplierMeasureByType, t=t)
        acceptance_prob *= beta_mult_measure.beta_factor(typ=self.site_dict[self.site_type[k]], t=t) \
            if beta_mult_measure else 1.0

        beta_mult_measure = self.measure_list.find(UpperBoundCasesBetaMultiplier, t=t)
        acceptance_prob *= beta_mult_measure.beta_factor(typ=self.site_dict[self.site_type[k]],
                                                         t=t,
                                                         t_pos_tests=self.t_pos_tests) \
            if beta_mult_measure else 1.0

        # return rejection prob
        rejection_prob = 1.0 - acceptance_prob
        return rejection_prob
    
    def is_person_home_from_visit_due_to_measure(self, t, i, visit_id, site_type):
        '''
        Returns True/False of whether person i stayed at home from visit
        `visit_id` due to any measures
        '''

        is_home = (
            self.measure_list.is_contained(
                SocialDistancingForAllMeasure, t=t,
                j=i, j_visit_id=visit_id) or 
            self.measure_list.is_contained(
                SocialDistancingBySiteTypeForAllMeasure, t=t,
                j=i, j_visit_id=visit_id, site_type=site_type) or
            self.measure_list.is_contained(
                SocialDistancingForPositiveMeasure, t=t,
                j=i, j_visit_id=visit_id, 
                state_posi_started_at=self.state_started_at['posi'],
                state_posi_ended_at=self.state_ended_at['posi'],
                state_resi_started_at=self.state_started_at['resi'],
                state_dead_started_at=self.state_started_at['dead']) or
            self.measure_list.is_contained(
                SocialDistancingByAgeMeasure, t=t,
                age=self.people_age[i], j_visit_id=visit_id) or
            self.measure_list.is_contained(
                SocialDistancingForSmartTracing, t=t,
                state_nega_started_at=self.state_started_at['nega'],
                state_nega_ended_at=self.state_ended_at['nega'],
                j=i, j_visit_id=visit_id) or 
            self.measure_list.is_contained(
                SocialDistancingSymptomaticAfterSmartTracing, t=t,
                state_isym_started_at=self.state_started_at['isym'],
                state_isym_ended_at=self.state_ended_at['isym'],
                state_nega_started_at=self.state_started_at['nega'],
                state_nega_ended_at=self.state_ended_at['nega'],
                j=i) or
            self.measure_list.is_contained(
                SocialDistancingForKGroups, t=t,
                j=i) or
            self.measure_list.is_contained(
                UpperBoundCasesSocialDistancing, t=t,
                j=i, j_visit_id=visit_id, t_pos_tests=self.t_pos_tests)
        )
        return is_home


    def __apply_for_testing(self, *, t, i, priority, trigger_tracing_if_positive):
        """
        Checks whether person i of should be tested and if so adds test to the testing queue
        """
        if t < self.testing_t_window[0] or t > self.testing_t_window[1]:
            return

        # fifo: first in, first out
        if self.test_queue_policy == 'fifo':
            self.testing_queue.push((i, trigger_tracing_if_positive), priority=t)

        # exposure-risk: has the following order of priority in queue:
        # 1) symptomatic tests, with `fifo` ordering (`priority = - max_time + t`)
        # 2) contact tracing tests: household members (`priority = 0.0`)
        # 3) contact tracing tests: contacts at sites (`priority` = lower empirical survival probability prioritized) 
        elif self.test_queue_policy == 'exposure-risk':
            self.testing_queue.push((i, trigger_tracing_if_positive), priority=priority)

        else:
            raise ValueError('Unknown queue policy')

    def __update_testing_queue(self, t):
        """
        Processes testing queue by popping the first `self.tests_per_batch` tests
        and adds `test` event (i.e. result) to event queue for person i with time lag `self.test_reporting_lag`
        """

        ctr = 0
        while (ctr < self.tests_per_batch) and (len(self.testing_queue) > 0):

            # get next individual to be tested
            ctr += 1
            i, trigger_tracing_if_positive = self.testing_queue.pop()

            # determine test result preemptively, to account for the individual's state at the time of testing
            if self.state['expo'][i] or self.state['ipre'][i] or self.state['isym'][i] or self.state['iasy'][i]:
                is_fn = np.random.binomial(1, self.test_fnr)
                if is_fn:
                    is_positive_test = False
                else:
                    is_positive_test = True
            else:
                is_fp = np.random.binomial(1, self.test_fpr)
                if is_fp:
                    is_positive_test = True
                else:
                    is_positive_test = False

            # push test result with delay to the event queue
            if t + self.test_reporting_lag < self.max_time:
                self.queue.push(
                    (t + self.test_reporting_lag, 'test', i, None, None, 
                     TestResult(is_positive_test=is_positive_test,
                                trigger_tracing_if_positive=trigger_tracing_if_positive)),
                    priority=t + self.test_reporting_lag)
                
            

    def __process_testing_event(self, t, i, metadata):
        """
        Processes return of test result of person `i` at time `t` with `metadata` from the event queue, which
        is a boolean indicator of a positive test result, which was collected at testing time, similar to the 
        blood sample of the person tested.
        """

        # extract `TestResult` data
        is_positive_test = metadata.is_positive_test
        trigger_tracing_if_positive = metadata.trigger_tracing_if_positive

        # collect test result based on "blood sample" taken before via `is_positive_test`
        # ... if positive
        if is_positive_test:

            # record timing only if tested positive for the first time
            if not self.state['posi'][i]:
                self.state_started_at['posi'][i] = t

            # mark as positive
            self.state['posi'][i] = True

            # mark as not negative
            if self.state['nega'][i]:
                self.state['nega'][i] = False
                self.state_ended_at['nega'][i] = t

        # ... if negative
        else:
            
            # record timing only if tested negative for the first time
            if not self.state['nega'][i]:
                self.state_started_at['nega'][i] = t

            # mark as negative
            self.state['nega'][i] = True
            
            # mark as not positive
            if self.state['posi'][i]:
                self.state['posi'][i] = False
                self.state_ended_at['posi'][i] = t

        # add timing of positive test for `UpperBoundCases` measures
        if is_positive_test:
            self.t_pos_tests.append(t)

        # if the individual is tested positive, process contact tracing when active and intended
        if self.state['posi'][i] and (self.smart_tracing_actions != []) and trigger_tracing_if_positive:
            self.__update_smart_tracing(t, i)
            self.__update_smart_tracing_housholds(t, i)
    
    def __update_smart_tracing(self, t, i):
        '''
        Updates smart tracing policy for individual `i` at time `t`.
        Iterates over possible contacts `j`
        '''

        # if i is generally not compliant: skip
        is_i_compliant = self.measure_list.is_compliant(
            ComplianceForAllMeasure, t=max(t - self.smart_tracing_contact_delta, 0.0), j=i)

        is_i_participating_in_manual_tracing = self.measure_list.is_active(
            ManualTracingForAllMeasure,
            t=max(t - self.smart_tracing_contact_delta, 0.0),
            j=i,
            j_visit_id=None)  # `None` indicates whether i is generally participating at all i.e. "non-visit-specific"

        if not (is_i_compliant or is_i_participating_in_manual_tracing):
            # no information available from `i`
            return 

        '''Find valid contacts of infector (excluding delta-contacts if beacons are not employed)'''
        infectors_contacts = self.mob.find_contacts_of_indiv(
            indiv=i,
            tmin=t - self.smart_tracing_contact_delta,
            tmax=t,
            tracing=True,
            p_reveal_visit=self.smart_tracing_p_willing_to_share)

        # filter which contacts were valid in dict keyed by individual
        valid_contacts_with_j = defaultdict(list)   
        for contact in infectors_contacts:            
            if self.__is_tracing_contact_valid(t=t, i=i, contact=contact):
                j = contact.indiv_i
                valid_contacts_with_j[j].append(contact)
        
        # if needed, compute empirical survival probability for all contacts
        if ('isolate' in self.smart_tracing_actions and self.smart_tracing_policy_isolate != 'basic') or \
           ('test' in self.smart_tracing_actions and self.smart_tracing_policy_test != 'basic'):

           # inspect whether the infector i was symptomatic or asymptomatic
            if self.state_started_at['iasy'][i] < np.inf:
                base_rate_i = self.mu
            else:
                base_rate_i = 1.0

            # compute empirical survival probability
            emp_survival_prob = dict()
            for j, contacts_j in valid_contacts_with_j.items():
                emp_survival_prob[j] = self.__compute_empirical_survival_probability(
                    t=t, i=i, j=j, base_rate=base_rate_i, contacts_i_j=contacts_j)

        '''Select contacts (not) to be traced based on tracing policy'''
        # each list contains (j, contacts_j) tuples, i.e. a part of `valid_contacts_with_j`
        # determining which valid individuals are traced 
        contacts_isolation, contacts_testing = [], []

        # isolation
        if 'isolate' in self.smart_tracing_actions:
            if self.smart_tracing_policy_isolate == 'basic':
                contacts_isolation, _ = self.__tracing_policy_basic(
                    contacts_with_j=valid_contacts_with_j, 
                    budget=self.smart_tracing_isolated_contacts)

            elif self.smart_tracing_policy_isolate == 'advanced':
                contacts_isolation, _ = self.__tracing_policy_advanced(
                    t=t, contacts_with_j=valid_contacts_with_j,
                    emp_survival_prob=emp_survival_prob, 
                    budget=self.smart_tracing_isolated_contacts)

            elif self.smart_tracing_policy_isolate == 'advanced-threshold':
                contacts_isolation, _ = self.__tracing_policy_advanced_threshold(
                    t=t, contacts_with_j=valid_contacts_with_j, 
                    threshold=self.smart_tracing_isolation_threshold,
                    emp_survival_prob=emp_survival_prob)
            else:
                raise ValueError('Invalid tracing isolation policy.')

        # testing
        if 'test' in self.smart_tracing_actions:
            
            if self.smart_tracing_policy_test == 'basic':
                contacts_testing, _ = self.__tracing_policy_basic(
                    contacts_with_j=valid_contacts_with_j, 
                    budget=self.smart_tracing_tested_contacts)

            elif self.smart_tracing_policy_test == 'advanced':
                contacts_testing, _ = self.__tracing_policy_advanced(
                    t=t, contacts_with_j=valid_contacts_with_j,
                    emp_survival_prob=emp_survival_prob, 
                    budget=self.smart_tracing_tested_contacts)

            elif self.smart_tracing_policy_test == 'advanced-threshold':
                contacts_testing, _ = self.__tracing_policy_advanced_threshold(
                    t=t, contacts_with_j=valid_contacts_with_j,
                    threshold=self.smart_tracing_testing_threshold,
                    emp_survival_prob=emp_survival_prob)
            else:
                raise ValueError('Invalid tracing test policy.')
            
        # record which contacts are being traced and which are not for later analysis
        self.__record_contacts_causing_trace_action(t=t, infector=i, contacts=valid_contacts_with_j)

        '''Execute contact tracing actions for selected contacts'''
        if 'isolate' in self.smart_tracing_actions:
            for j, _ in contacts_isolation:
                self.measure_list.start_containment(SocialDistancingForSmartTracing, t=t, j=j)
                self.measure_list.start_containment(SocialDistancingForSmartTracingHousehold, t=t, j=j)
                self.measure_list.start_containment(SocialDistancingSymptomaticAfterSmartTracing, t=t, j=j)
                self.measure_list.start_containment(SocialDistancingSymptomaticAfterSmartTracingHousehold, t=t, j=j)

        if 'test' in self.smart_tracing_actions:
            for j, _ in contacts_testing:
                if self.smart_tracing_policy_test == 'basic':
                    self.__apply_for_testing(t=t, i=j, priority=1.0, 
                        trigger_tracing_if_positive=self.trigger_tracing_after_posi_trace_test)
                elif self.smart_tracing_policy_test == 'advanced' \
                    or self.smart_tracing_policy_test == 'advanced-threshold':
                    self.__apply_for_testing(t=t, i=j, priority=emp_survival_prob[j], 
                        trigger_tracing_if_positive=self.trigger_tracing_after_posi_trace_test)
                else:
                    raise ValueError('Invalid smart tracing policy.')

    def __update_smart_tracing_housholds(self, t, i):
        '''Execute contact tracing actions for _household members_'''
        for j in self.households[self.people_household[i]]:

            if self.state['dead'][j]:
                continue

            # contact tracing action
            if 'isolate' in self.smart_tracing_actions:
                self.measure_list.start_containment(SocialDistancingForSmartTracing, t=t, j=j)
                self.measure_list.start_containment(SocialDistancingForSmartTracingHousehold, t=t, j=j)
                self.measure_list.start_containment(SocialDistancingSymptomaticAfterSmartTracing, t=t, j=j)
                self.measure_list.start_containment(SocialDistancingSymptomaticAfterSmartTracingHousehold, t=t, j=j)

            if 'test' in self.smart_tracing_actions:
                # don't test positive people twice
                if not self.state['posi'][j]:
                    # household members always treated as `empirical survival prob. = 0` for `exposure-risk` policy
                    # not relevant for `fifo` queue
                    self.__apply_for_testing(t=t, i=j, priority=0.0,
                        trigger_tracing_if_positive=self.trigger_tracing_after_posi_trace_test)

    def __tracing_policy_basic(self, contacts_with_j, budget):
        """
        Basic contact tracing. Selects random contacts up to the limit.
        `contacts_with_j`:   {j : contacts} where `contacts` are contacts of infector with j 
                             in the contact tracing time window
        """
        # randomly permute all (j, contacts_with_j) tuples
        n = len(contacts_with_j)
        js = list(contacts_with_j.keys())
        contact_with_js = list(contacts_with_j.values())
        p = np.random.permutation(n).tolist()
        tuples = list(zip([js[p[i]] for i in range(n)],
                          [contact_with_js[p[i]] for i in range(n)]))

        # return (traced, not traced)
        return tuples[:budget], tuples[budget:]

    def __tracing_policy_advanced(self, t, contacts_with_j, budget, emp_survival_prob):
        """
        Advanced contact tracing. Selects contacts according to high exposure risk up to the limit.
        `contacts_with_j`:   {j : contacts} where `contacts` are contacts of infector with j 
                             in the contact tracing time window
        `emp_survival_prob`: {j : empirical probability of j not being infected}
        """

        # sort by empirical probability of survival (lowest first)
        p = np.array([emp_survival_prob[j] for j in contacts_with_j.keys()]).argsort().tolist()
        n = len(contacts_with_j)
        js = list(contacts_with_j.keys())
        contact_with_js = list(contacts_with_j.values())
        p = np.random.permutation(n).tolist()
        tuples = list(zip([js[p[i]] for i in range(n)],
                          [contact_with_js[p[i]] for i in range(n)]))

        # return (traced, not traced)
        return tuples[:budget], tuples[budget:]

    def __tracing_policy_advanced_threshold(self, t, contacts_with_j, threshold, emp_survival_prob):
        """
        Advanced contact tracing. Selects contacts that have higher exposure risk than the threshold.
        `contacts_with_j`:   {j : contacts} where `contacts` is list of contacts of infector with j 
                             in the contact tracing time window
        `emp_survival_prob`: {j : empirical probability of j not being infected}
        """
        # only trace above a certain empirical probability of exposure 
        traced =     [(j, contact_with_j) for j, contact_with_j in contacts_with_j.items() if (1 - emp_survival_prob[j]) >  threshold]
        not_traced = [(j, contact_with_j) for j, contact_with_j in contacts_with_j.items() if (1 - emp_survival_prob[j]) <= threshold]

        # return (traced, not traced)
        return traced, not_traced
          

    def __record_contacts_causing_trace_action(self, *, t, infector, contacts):
        """
        Records the contact tuples `contacts` (j, contacts of j with infector) that ended up
        triggering or not triggering a tracing action `action` for individual `j`.
        Used to later compute true positives, false positives, etc.
        """
        if len(self.thresholds_roc) > 0:

            if (self.smart_tracing_stats_window[0] <= t) and \
               (self.smart_tracing_stats_window[1] > t):

                self.valid_contacts_for_tracing.append((t, infector, contacts))


    def __is_tracing_contact_valid(self, *, t, i, contact):
        """ 
        Compute whether a contact of individual i at time t is valid
        This is called with `i` being the infector.
        """

        start_contact = contact.t_from
        j = contact.indiv_i
        site_id = contact.site
        j_visit_id, i_visit_id = contact.id_tup
        site_type = self.mob.site_dict[self.mob.site_type[site_id]]


        '''Check status of both individuals'''
        i_has_valid_status = (
            # not dead
            (not (self.state['dead'][i] and self.state_started_at['dead'][i] <= start_contact)) and

            # not hospitalized at time of contact
            (not (self.state['hosp'][i] and self.state_started_at['hosp'][i] <= start_contact))
        )

        j_has_valid_status = (
            # not dead
            (not (self.state['dead'][j] and self.state_started_at['dead'][j] <= start_contact)) and

            # not hospitalized at time of contact
            (not (self.state['hosp'][j] and self.state_started_at['hosp'][j] <= start_contact)) and

            # not positive at time of tracing
            (not (self.state['posi'][j] and self.state_started_at['posi'][j] <= t))
        )

        if (not i_has_valid_status) or (not j_has_valid_status):
            return False
        
        '''Check compliance'''
        is_i_compliant = self.measure_list.is_compliant(
            ComplianceForAllMeasure, 
            # to be consistent with general `is_i_compliant` check outside, don't use `start_contact`
            t=max(t - self.smart_tracing_contact_delta, 0.0), j=i)
            
        if is_i_compliant:
            is_i_compliant_or_recalls = True
        else:
            # if infector `i` not compliant with standard tracing, a contact can still be identified when
            # (i) `i` participates in manual tracing, and either
            # (iia) a bluetooth beacon is at the site
            # (iib) or `j` is reachable over offline manual tracing

            # check if `i` recalls the visit and hence its site
            i_recalls_visit = self.measure_list.is_active(
                ManualTracingForAllMeasure,
                t=start_contact,  # t not needed for the visit, but only for whether measure is active
                j=i,
                j_visit_id=i_visit_id)  # `i_visit_id` queries whether `i` recalls this specific visit

            is_i_compliant_or_recalls = i_recalls_visit

        # 1) j can be traced with digital tracing
        is_j_compliant = self.measure_list.is_compliant(
            ComplianceForAllMeasure, 
            # to be consistent with `is_i_compliant` check, don't use `start_contact`
            t=max(t - self.smart_tracing_contact_delta, 0.0), j=j)

        # 2) j can be traced with offline manual tracing
        is_j_manually_tracable = self.measure_list.is_active(
            ManualTracingReachabilityForAllMeasure,
            # to be consistent with `is_i_compliant` check, don't use `start_contact`
            t=max(t - self.smart_tracing_contact_delta, 0.0), j=j,
            j_visit_id=j_visit_id,
            site_type=site_type)

        # If either i or j does not comply with any tracing method (digital/offline)
        if (not is_i_compliant_or_recalls) or (not (is_j_compliant or is_j_manually_tracable)):
            return False

        '''Check that site traces contact'''
        if not is_j_manually_tracable and self.mob.beacon_config is not None:
            if not self.mob.site_has_beacon[site_id]:
                return False

        '''Check SocialDistancing measures'''
        is_i_contained = self.is_person_home_from_visit_due_to_measure(
            t=start_contact, i=i, visit_id=i_visit_id, 
            site_type=self.site_dict[self.site_type[site_id]])
        is_j_contained = self.is_person_home_from_visit_due_to_measure(
            t=start_contact, i=j, visit_id=j_visit_id, 
            site_type=self.site_dict[self.site_type[site_id]])

        if is_i_contained or is_j_contained:
            return False

        # if all of the above checks passed, then contact is valid
        return True

    def __compute_empirical_survival_probability(self, *, t, i, j, contacts_i_j, base_rate=1.0, ignore_sites=False):
        """ Compute empirical survival probability of individual j due to node i at time t"""
        
        s = 0

        for contact in contacts_i_j:

            t_start = contact.t_from
            t_end = contact.t_to
            t_end_direct = contact.t_to_direct
            site = contact.site

            # break if next contact starts after t
            if t_start >= t:
                break

            # check whether this computation has access to site information
            if self.mob.site_has_beacon[site] and not ignore_sites:
                s += self.__survival_prob_contribution_with_site(
                    i=i, j=j, site=site, t=t, t_start=t_start, 
                    t_end_direct=t_end_direct, t_end=t_end, base_rate=base_rate)
            else:
                s += self.__survival_prob_contribution_no_site(
                    t=t, t_start=t_start, t_end_direct=t_end_direct, base_rate=base_rate)

        # survival probability
        survival_prob = np.exp(-s)
        return survival_prob

    def __survival_prob_contribution_no_site(self, *, t, t_start, t_end_direct, base_rate):
        """Computes empirical survival probability estimate when no site information
            such as non-contemporaneous contact is known.

            t:             time of tracing action (upper bound on visit time)
            t_start:       start of contact
            t_end_direct:  end of direct contact
        """

        # only consider direct contact
        if min(t_end_direct, t) >= t_start:
            # assume infector was at site entire `delta` time window 
            # before j arrived by lack of information otherwise
            return (min(t_end_direct, t) - t_start) * base_rate * self.betas_weighted_mean * self.__kernel_term(- self.delta, 0.0, 0.0)
        else:
            return 0.0

    def __survival_prob_contribution_with_site(self, *, i, j, site, t, t_start, t_end_direct, t_end, base_rate):
        """Computes exact empirical survival probability estimate when site information
            such as site-specific transmission rate and 
            such as non-contemporaneous contact is known.

            i:              infector
            j:              individual at risk due to `i`
            site:           site
            t:              time of tracing action (upper bound on visit time)
            t_start:        start of contact
            t_end_direct:   end of direct contact
            t_end:          end of contact
            base_rate:      base rate
        """

        # query visit of infector i that resulted in the contact
        inf_visit_ = list(self.mob.list_intervals_in_window_individual_at_site(
            indiv=i, site=site, t0=t_end_direct, t1=t_end_direct))
        assert(len(inf_visit_) == 1)
        inf_from, inf_to = inf_visit_[0].left, inf_visit_[0].right

        # query visit of j that resulted in the contact
        j_visit_ = list(self.mob.list_intervals_in_window_individual_at_site(
            indiv=j, site=site, t0=t_start, t1=t_start))
        assert(len(j_visit_) == 1)
        j_from, j_to = j_visit_[0].left, j_visit_[0].right

        # BetaMultiplier measures
        beta_fact = 1.0
        beta_mult_measure = self.measure_list.find(BetaMultiplierMeasureBySite, t=t_start)
        beta_fact *= beta_mult_measure.beta_factor(k=site, t=t_start) \
            if beta_mult_measure else 1.0
        
        beta_mult_measure = self.measure_list.find(BetaMultiplierMeasureByType, t=t_start)
        beta_fact *= (beta_mult_measure.beta_factor(typ=self.site_dict[self.site_type[site]], t=t_start)
            if beta_mult_measure else 1.0)

        beta_mult_measure = self.measure_list.find(UpperBoundCasesBetaMultiplier, t=t)
        beta_fact *= (beta_mult_measure.beta_factor(typ=self.site_dict[self.site_type[site]], t=t, t_pos_tests=self.t_pos_tests) \
            if beta_mult_measure else 1.0)

        # contact contribution
        expo_int = self.exposure_integral(
            j_from=j_from,
            j_to=min(j_to, t),
            inf_from=inf_from,
            inf_to=min(inf_to, t),
            beta_site=beta_fact * self.betas[self.site_dict[self.site_type[site]]],
            base_rate=base_rate,
        )
        return expo_int
