
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

from lib.priorityqueue import PriorityQueue
from lib.measures import (MeasureList, BetaMultiplierMeasure,
    SocialDistancingForAllMeasure, BetaMultiplierMeasureByType,
    SocialDistancingForPositiveMeasure, SocialDistancingByAgeMeasure, 
    SocialDistancingForSmartTracing, ComplianceForAllMeasure, SocialDistancingForKGroups)

class DiseaseModel(object):
    """
    Simulate continuous-time SEIR epidemics with exponentially distributed inter-event times.
    All units in the simulator are in hours for numerical stability, though disease parameters are
    assumed to be in units of days as usual in epidemiology
    """

    def __init__(self, mob, distributions, dynamic_tracing=False):
        """
        Init simulation object with parameters

        Arguments:
        ---------
        mob:
            object of class MobilitySimulator providing mobility data

        dynamic_tracing: bool
            If true contacts are computed on-the-fly during launch_simulation
            instead of using the previously filled contact array

        """

        # cache settings
        self.mob = mob
        self.d = distributions
        self.dynamic_tracing = dynamic_tracing

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
        self.num_site_types = mob.num_site_types
        
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
        
        # smart tracing
        self.empirical_survival_probability = np.ones(self.n_people, dtype='float')

    def initialize_states_for_seeds(self):
        """
        Sets state variables according to invariants as given by `self.initial_seeds`
        """
        assert(isinstance(self.initial_seeds, dict))
        for state, seeds_ in self.initial_seeds.items():
            for i in seeds_:
                assert(self.was_initial_seed[i] == False)
                self.was_initial_seed[i] = True
                
                # inital exposed
                if state == 'expo':
                    self.__process_exposure_event(0.0, i, None)

                # initial presymptomatic
                elif state == 'ipre':
                    self.state['susc'][i] = False
                    self.state['expo'][i] = True

                    self.state_ended_at['susc'][i] = 0.0
                    self.state_started_at['expo'][i] = 0.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_presymptomatic_event(0.0, i)


                # initial asymptomatic
                elif state == 'iasy':
                    self.state['susc'][i] = False
                    self.state['expo'][i] = True

                    self.state_ended_at['susc'][i] = 0.0
                    self.state_started_at['expo'][i] = 0.0

                    self.bernoulli_is_iasy[i] = 1
                    self.__process_asymptomatic_event(0.0, i)

                # initial symptomatic
                elif state == 'isym' or state == 'isym_notposi':

                    self.state['susc'][i] = False
                    self.state['ipre'][i] = True

                    self.state_ended_at['susc'][i] = 0.0
                    self.state_started_at['expo'][i] = 0.0
                    self.state_ended_at['expo'][i] = 0.0
                    self.state_started_at['ipre'][i] = 0.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__push_contact_exposure_events(0.0, i, 1.0)
                    self.__process_symptomatic_event(0.0, i)

                # initial symptomatic and positive
                elif state == 'isym_posi':

                    self.state['susc'][i] = False
                    self.state['ipre'][i] = True
                    self.state['posi'][i] = True

                    self.state_ended_at['susc'][i] = 0.0
                    self.state_started_at['expo'][i] = 0.0
                    self.state_ended_at['expo'][i] = 0.0
                    self.state_started_at['ipre'][i] = 0.0
                    self.state_started_at['posi'][i] = 0.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__push_contact_exposure_events(0.0, i, 1.0)
                    self.__process_symptomatic_event(0.0, i)

                # initial resistant and positive
                elif state == 'resi_posi':

                    self.state['susc'][i] = False
                    self.state['isym'][i] = True
                    self.state['posi'][i] = True

                    self.state_ended_at['susc'][i] = 0.0
                    self.state_started_at['expo'][i] = 0.0
                    self.state_ended_at['expo'][i] = 0.0
                    self.state_started_at['ipre'][i] = 0.0
                    self.state_ended_at['ipre'][i] = 0.0
                    self.state_started_at['isym'][i] = 0.0
                    self.state_started_at['posi'][i] = 0.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_resistant_event(0.0, i)

                # initial resistant and positive
                elif state == 'resi_notposi':

                    self.state['susc'][i] = False
                    self.state['isym'][i] = True

                    self.state_ended_at['susc'][i] = 0.0
                    self.state_started_at['expo'][i] = 0.0
                    self.state_ended_at['expo'][i] = 0.0
                    self.state_started_at['ipre'][i] = 0.0
                    self.state_ended_at['ipre'][i] = 0.0
                    self.state_started_at['isym'][i] = 0.0

                    self.bernoulli_is_iasy[i] = 0
                    self.__process_resistant_event(0.0, i)

                else:
                    raise ValueError('Invalid initial seed state.')

    def launch_epidemic(self, params, initial_counts, testing_params, measure_list, verbose=True):
        """
        Run the epidemic, starting from initial event list.
        Events are treated in order in a priority queue. An event in the queue is a tuple
        the form
            `(time, event_type, node, infector_node, location)`

        """
        self.verbose = verbose

        # optimized params
        self.betas = params['betas']
        self.alpha = params['alpha']
        self.mu = params['mu']

        # testing settings
        self.testing_frequency  = testing_params['testing_frequency']
        self.test_targets       = testing_params['test_targets']
        self.test_queue_policy  = testing_params['test_queue_policy']
        self.test_reporting_lag = testing_params['test_reporting_lag']        
        self.tests_per_batch    = testing_params['tests_per_batch']
        self.testing_t_window   = testing_params['testing_t_window']
        
        # smart tracing
        self.smart_tracing       = testing_params['smart_tracing']
        self.test_smart_action   = testing_params['test_smart_action']
        self.test_smart_delta    = testing_params['test_smart_delta']
        self.test_smart_num_contacts   = testing_params['test_smart_num_contacts']
        self.test_smart_duration = testing_params['test_smart_duration']
        
        # Set list of measures
        if not isinstance(measure_list, MeasureList):
            raise ValueError("`measure_list` must be a `MeasureList` object")
        self.measure_list = measure_list

        # Sample bernoulli outcome for all SocialDistancingForAllMeasure
        self.measure_list.init_run(SocialDistancingForAllMeasure,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))

        self.measure_list.init_run(SocialDistancingForPositiveMeasure,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))

        self.measure_list.init_run(SocialDistancingByAgeMeasure,
                                   num_age_groups=self.num_age_groups,
                                   n_visits=max(self.mob.visit_counts))
        
        self.measure_list.init_run(ComplianceForAllMeasure,
                                   n_people=self.n_people)
        
        self.measure_list.init_run(SocialDistancingForSmartTracing,
                                   n_people=self.n_people,
                                   n_visits=max(self.mob.visit_counts))    

        self.measure_list.init_run(SocialDistancingForKGroups)

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

        ### sample all iid events ahead of time in batch
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
            delta_susc_to_expo = self.d.sample_susc_baseexpo(size=self.n_people)
            for i in range(self.n_people):
                if not self.was_initial_seed[i]:
                    # sample non-contact exposure events
                    self.queue.push(
                        (delta_susc_to_expo[i], 'expo', i, None, None),
                        priority=delta_susc_to_expo[i])

        # initialize test processing events: add 'update_test' event to queue for `testing_frequency` hour
        for h in range(1, math.floor(self.max_time / self.testing_frequency)):
            ht = h * self.testing_frequency
            self.queue.push((ht, 'execute_tests', None, None, None), priority=ht)

        # MAIN EVENT LOOP
        t = 0.0
        while self.queue:

            # get next event to process
            t, event, i, infector, k = self.queue.pop()

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
                i_susceptible = ((not self.state['expo'][i])
                                    and (self.state['susc'][i]))

                # base rate exposure
                if (infector is None) and i_susceptible:
                    self.__process_exposure_event(t, i, None)

                # contact exposure
                if (infector is not None) and i_susceptible:

                    is_in_contact, contact = self.mob.is_in_contact(indiv_i=i, indiv_j=infector, site=k, t=t)
                    assert(is_in_contact and (k is not None))
                    i_visit_id, infector_visit_id = contact.id_tup

                    # 1) check whether infector recovered or dead
                    infector_recovered = \
                        (self.state['resi'][infector] or 
                            self.state['dead'][infector])

                    # 2) check whether infector stayed at home due to measures
                    #    or got hospitalized
                    infector_contained = self.is_person_home_from_visit_due_to_measure(
                        t=t, i=infector, visit_id=infector_visit_id) \
                        or self.state['hosp'][infector]
                                            
                    # 3) check whether susceptible stayed at home due to measures
                    i_contained = self.is_person_home_from_visit_due_to_measure(
                        t=t, i=i, visit_id=i_visit_id)  

                    # 4) check whether infectiousness got reduced due to site specific 
                    #    measures and as a consequence this event didn't occur
                    rejection_prob = self.reject_exposure_due_to_measure(t=t, k=k)
                    site_avoided_infection =  (np.random.uniform() < rejection_prob)

                    # if none of 1), 2), 3), 4) are true, the event is valid
                    if  (not infector_recovered) and \
                        (not infector_contained) and \
                        (not i_contained) and \
                        (not site_avoided_infection):

                        self.__process_exposure_event(t, i, infector)

                    # if any of 2), 3), 4) were true, an infection could happen 
                    # at a later point, hence sample a new event 
                    if (infector_contained or i_contained or site_avoided_infection):

                        mu_infector = self.mu if self.state['iasy'][infector] else 1.0
                        self.__push_contact_exposure_infector_to_j(
                            t=t, infector=infector, j=i, base_rate=mu_infector)                    

            elif event == 'ipre':
                self.__process_presymptomatic_event(t, i)

            elif event == 'iasy':
                self.__process_asymptomatic_event(t, i)

            elif event == 'isym':
                self.__process_symptomatic_event(t, i)

            elif event == 'resi':
                self.__process_resistant_event(t, i)

            elif event == 'test':
                self.__process_testing_event(t, i)

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

        # free memory
        del self.queue

    def __process_exposure_event(self, t, i, parent):
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
            self.queue.push(
                (t + self.delta_expo_to_iasy[i], 'iasy', i, None, None),
                priority=t + self.delta_expo_to_iasy[i])
        else:
            self.queue.push(
                (t + self.delta_expo_to_ipre[i], 'ipre', i, None, None),
                priority=t + self.delta_expo_to_ipre[i])

    def __process_presymptomatic_event(self, t, i):
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

        # resistant event
        self.queue.push(
            (t + self.delta_ipre_to_isym[i], 'isym', i, None, None),
            priority=t + self.delta_ipre_to_isym[i])

        # contact exposure of others
        self.__push_contact_exposure_events(t, i, 1.0)

    def __process_symptomatic_event(self, t, i):
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
        if self.test_targets == 'isym':
            self.__apply_for_testing(t, i)

        # hospitalized?
        if self.bernoulli_is_hospi[i]:
            self.queue.push(
                (t + self.delta_isym_to_hosp[i], 'hosp', i, None, None),
                priority=t + self.delta_isym_to_hosp[i])

        # resistant event vs fatality event
        if self.bernoulli_is_fatal[i]:
            self.queue.push(
                (t + self.delta_isym_to_dead[i], 'dead', i, None, None),
                priority=t + self.delta_isym_to_dead[i])
        else:
            self.queue.push(
                (t + self.delta_isym_to_resi[i], 'resi', i, None, None),
                priority=t + self.delta_isym_to_resi[i])

    def __process_asymptomatic_event(self, t, i):
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
        self.queue.push(
            (t + self.delta_iasy_to_resi[i], 'resi', i, None, None),
            priority=t + self.delta_iasy_to_resi[i])

        # contact exposure of others
        self.__push_contact_exposure_events(t, i, self.mu)

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
        \int_a^b exp(self.gamma * (u - T)) du
        =  exp(- self.gamma * T) (exp(self.gamma * b) - exp(self.gamma * a)) / self.gamma
        '''
        return (np.exp(self.gamma * (b - T)) - np.exp(self.gamma * (a - T))) / self.gamma


    def __push_contact_exposure_events(self, t, infector, base_rate):
        """
        Pushes all exposure events that person `i` causes
        for other people via contacts, using `base_rate` as basic infectivity
        of person `i` (equivalent to `\mu` in model definition)
        """

        if not self.dynamic_tracing:
            def valid_j():
                '''Generates indices j where `infector` is present
                at least `self.delta` hours before j '''
                for j in range(self.n_people):
                    if self.state['susc'][j]:
                        if self.mob.will_be_in_contact(indiv_i=j, indiv_j=infector, t=t, site=None):
                            yield j

            valid_contacts = valid_j()
        else:
            infectors_contacts = self.mob.find_contacts_of_indiv(self.mob.all_mob_traces, indiv=infector, tmin=t)

            valid_contacts = []
            for contact in infectors_contacts:
                if self.state['susc'][contact.indiv_i]:
                    if contact not in self.mob.contacts[contact.indiv_i][infector]:
                        self.mob.contacts[contact.indiv_i][infector].update([contact])
                    if contact.indiv_i not in valid_contacts:
                        valid_contacts.append(contact.indiv_i)

        # generate potential exposure event for `j` from contact with `infector`
        for j in valid_contacts:
            self.__push_contact_exposure_infector_to_j(t=t, infector=infector, j=j, base_rate=base_rate)



    def __push_contact_exposure_infector_to_j(self, t, infector, j, base_rate):
        """
        Pushes all the next exposure event that person `infector` causes for person `j`
        using `base_rate` as basic infectivity of person `i` 
        (equivalent to `\mu` in model definition)
        """
        tau = t
        sampled_event = False
        Z = self.__kernel_term(- self.delta, 0.0, 0.0)

        # sample next arrival from non-homogeneous point process
        while self.mob.will_be_in_contact(indiv_i=j, indiv_j=infector, t=tau, site=None) and not sampled_event:
            
            # check if j could get infected from infector at current `tau`
            # i.e. there is `delta`-contact from infector to j (i.e. non-zero intensity)
            has_infectious_contact, contact = self.mob.is_in_contact(indiv_i=j, indiv_j=infector, t=tau, site=None)

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
            lambda_max = max(self.betas) * base_rate * Z
            tau += 24.0 * np.random.exponential(scale=1.0 / lambda_max)

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
            beta_k = self.betas[self.site_type[site]]
            p = (beta_k * base_rate * sum([self.__kernel_term(v[0], v[1], tau) for v in intersections])) \
                / lambda_max
            
            assert(p <= 1 + 1e-8 and p >= 0)

            # accept w.prob. lambda(t) / lambda_max
            u = np.random.uniform()
            if u <= p:
                self.queue.push(
                    (tau, 'expo', j, infector, site), priority=tau)
                sampled_event = True


    def reject_exposure_due_to_measure(self, t, k):
        '''
        Returns rejection probability of exposure event not occuring
        at location k at time k
        Searches through BetaMultiplierMeasures and retrieves beta multipliers
        Scaling beta is equivalent to scaling down the acceptance probability
        '''

        acceptance_prob = 1.0

        # BetaMultiplierMeasures
        beta_mult_measure = self.measure_list.find(BetaMultiplierMeasure, t=t)
        acceptance_prob *= beta_mult_measure.beta_factor(k=k, t=t) if beta_mult_measure else 1.0

        beta_mult_measure = self.measure_list.find(BetaMultiplierMeasureByType, t=t)
        acceptance_prob *= beta_mult_measure.beta_factor(typ=self.site_type[k], t=t) if beta_mult_measure else 1.0

        # return rejection prob
        rejection_prob = 1.0 - acceptance_prob
        return rejection_prob
    
    def is_person_home_from_visit_due_to_measure(self, t, i, visit_id):
        '''
        Returns True/False of whether person i stayed at home from visit
        `visit_id` due to any measures
        '''

        is_home = (
            self.measure_list.is_contained(
                SocialDistancingForAllMeasure, t=t,
                j=i, j_visit_id=visit_id) or 
            self.measure_list.is_contained(
                SocialDistancingForPositiveMeasure, t=t,
                j=i, j_visit_id=visit_id, state_posi=self.state['posi'], state_resi=self.state['resi'], state_dead=self.state['dead']) or 
            self.measure_list.is_contained(
                SocialDistancingByAgeMeasure, t=t,
                age=self.people_age[i], j_visit_id=visit_id) or
            self.measure_list.is_contained(
                SocialDistancingForSmartTracing, t=t,
                j=i, j_visit_id=visit_id) or 
            self.measure_list.is_contained(
                SocialDistancingForKGroups, t=t,
                j=i)            
        )
        return is_home


    def __apply_for_testing(self, t, i, s=0.0):
        """
        Checks whether person i of should be tested and if so adds test to the testing queue
        """
        if t < self.testing_t_window[0] or t > self.testing_t_window[1]:
            return

        # fifo: priority = current time
        if self.test_queue_policy == 'fifo':
            self.testing_queue.push(i, priority=t)
        else:
            raise ValueError('Unknown queue policy')

    def __update_testing_queue(self, t):
        """
        Processes testing queue by popping the first `self.tests_per_batch` tests
        and adds `test` event to event queue for person i with time lag `self.test_reporting_lag`
        """

        ctr = 0
        while (ctr < self.tests_per_batch) and (len(self.testing_queue) > 0):
            ctr += 1
            i = self.testing_queue.pop()
            self.queue.push((t + self.test_reporting_lag, 'test',
                                i, None, None), priority=t + self.test_reporting_lag)
            
            # update test result preemptively, to account for the state at the time of testing
            if self.state['expo'][i] or self.state['ipre'][i] or self.state['isym'][i] or self.state['iasy'][i]:
                self.outcome_of_test[i] = True
            else:
                self.outcome_of_test[i] = False

    def __process_testing_event(self, t, i):
        """
        Test person `i` at time `t`
        """
        
        # update test result preemptively, to account for the state at the time of testing
        if self.outcome_of_test[i]: 
            self.state['posi'][i] = True
            self.state_started_at['posi'][i] = t
                
            if self.state['nega'][i]:
                self.state['nega'][i] = False
                self.state_ended_at['nega'][i] = self.state_started_at['posi'][i]
        else:
            self.state['nega'][i] = True
            self.state_started_at['nega'][i] = t
            
        # smart tracing
        # if i is not compliant, skip
        is_i_not_compliant = self.measure_list.is_contained(
            ComplianceForAllMeasure, t=t-self.test_smart_delta, j=i)
            
        if is_i_not_compliant:
            return
        
        if self.state['posi'][i] and (self.smart_tracing != None):
            self.__update_smart_tracing(t, i)
    
    def __update_smart_tracing(self, t, i):
        '''
        Updates smart tracing policy for individual `i` at time `t`.
        Iterates over possible contacts `j`

        '''
        if not self.dynamic_tracing:
            def valid_j():
                '''Generate individuals j where `i` was present
                up to `self.test_smart_delta` hours before t '''
                for j in range(self.n_people):
                    if not self.state['dead'][j]:
                        if self.mob.will_be_in_contact(indiv_i=j, indiv_j=i, site=None, t=t-self.test_smart_delta):
                            yield j

            valid_contacts = valid_j()
        else:
            infectors_contacts = self.mob.find_contacts_of_indiv(self.mob.all_mob_traces, indiv=i,
                                                                 tmin=t - self.test_smart_delta)
            valid_contacts = []

            for contact in infectors_contacts:
                if not self.state['dead'][contact.indiv_i]:
                    if contact not in self.mob.contacts[contact.indiv_i][i]:
                        self.mob.contacts[contact.indiv_i][i].update([contact])
                    if contact.indiv_i not in valid_contacts:
                        valid_contacts.append(contact.indiv_i)

        contacts = PriorityQueue()
        
        for j in valid_contacts:
            # if j is not compliant, skip
            is_j_not_compliant = self.measure_list.is_contained(
                ComplianceForAllMeasure, t=t-self.test_smart_delta, j=j)
            
            if is_j_not_compliant:
                continue

            valid_contact, s = self.__compute_empirical_survival_probability(t, i, j)
            
            if valid_contact:
                self.empirical_survival_probability[j] = s
                if self.smart_tracing == 'basic':
                    contacts.push(j, priority=t)
                elif self.smart_tracing == 'advanced':
                    contacts.push(j, priority=self.empirical_survival_probability[j])
                else:
                    raise ValueError('Invalid smart tracing policy.')
        
        # quarantine nodes for a 'self.test_smart_duration'
        max_contacts = len(contacts)
        for j in range(min(self.test_smart_num_contacts, max_contacts)):
            contact = contacts.pop()
            if self.test_smart_action == 'isolate':
                self.measure_list.start_containment(SocialDistancingForSmartTracing, t=t, j=contact)
            if self.test_smart_action == 'test':
                self.__apply_for_testing(t, contact)
    
    # compute empirical survival probability of individual j due to node i at time t
    def __compute_empirical_survival_probability(self, t, i, j):
        s = 0
        valid_contact = False
            
        next_contact_obj = self.mob.next_contact(indiv_i=j, indiv_j=i, t=t - self.test_smart_delta, site=None)      
        while next_contact_obj is not None:

            start_next_contact = next_contact_obj.t_from
            end_next_contact = next_contact_obj.t_to

            # break if next contact is >= t
            if start_next_contact >= t:
                break

            is_in_contact, contact = self.mob.is_in_contact(indiv_i=j, indiv_j=i, site=None, t=start_next_contact)
            assert(is_in_contact)
            j_visit_id, i_visit_id = contact.id_tup
                    
            # Check SocialDistancing measures
            is_j_contained = self.is_person_home_from_visit_due_to_measure(t=start_next_contact, i=j, visit_id=j_visit_id)  
            is_i_contained = self.is_person_home_from_visit_due_to_measure(t=start_next_contact, i=i, visit_id=i_visit_id)
                
            # check hospitalization
            is_i_contained = is_i_contained or (
                self.state['hosp'][i] and self.state_started_at['hosp'][i] < start_next_contact)
                    
            # BetaMultiplier measures
            site = contact.site
            beta_fact = 1.0

            beta_mult_measure = self.measure_list.find(BetaMultiplierMeasure, t=start_next_contact)
            beta_fact *= beta_mult_measure.beta_factor(k=site, t=start_next_contact) if beta_mult_measure else 1.0
            
            beta_mult_measure = self.measure_list.find(BetaMultiplierMeasureByType, t=start_next_contact)
            beta_fact *= beta_mult_measure.beta_factor(typ=self.site_type[site], t=start_next_contact) if beta_mult_measure else 1.0 
                
            # decide if i and j really had overlap
            if (not is_j_contained) and (not is_i_contained):
                if self.smart_tracing == 'basic':
                    valid_contact = True
                    break
                elif self.smart_tracing == 'advanced':
                    s += (min(end_next_contact, t) - start_next_contact) * self.betas[self.site_type[site]] * beta_fact
                    valid_contact = True
                
            # get next contact (if it exists)
            next_contact_obj = self.mob.next_contact(indiv_i=j, indiv_j=i, t=end_next_contact + self.delta, site=None)
        
        s = np.exp(-s)
        
        return valid_contact, s
