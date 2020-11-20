from collections import namedtuple, defaultdict
import itertools
import random as rd
import pandas as pd
import numpy as np
import numba
import pickle
import json

from interlap import InterLap

TO_HOURS = 24.0

# Tuple representing a vist of an individual at a site
# Note: first two elements must be('t_from', 't_to_shifted') to match contacts using `interlap`
Visit = namedtuple('Visit', (
    't_from',        # Time of arrival at site
    't_to_shifted',  # Time influence of visit ends (i.e. time of departure, shifted by `delta`)
    't_to',          # Time of departure from site
    'indiv',         # Id of individual
    'site',          # Id of site
    'duration',      # Duration of visit (i.e. `t_to` - `t_from`)
    'id'             # unique id of visit, used to identify specific visits of `indiv`
))

# Tupe representing a contact from a individual i to another individual j
# where individual i is at risk due to j
Contact = namedtuple('Contact', (
    't_from',   # Time of beginning of contact
    't_to',     # Time of end of contact
    'indiv_i',  # Id of individual 'from' contact (uses interval (`t_from`, `t_to`) for matching)
    'indiv_j',  # Id of individual 'to' contact (may have already left, uses interval (`t_from`, `t_to_shifted`) for matching)
    'site',     # Id of site
    'duration', # Duration of contact (i.e. when i was at risk due to j)
    'id_tup'    # tuple of `id`s of visits of `indiv_i` and `indiv_j`
))

# Tuple representing an interval for back-operability with previous version
# using pandas.Interval objects
Interval = namedtuple('Interval', ('left', 'right'))

@numba.njit
def _simulate_individual_synthetic_trace(indiv, num_sites, max_time, home_loc, site_loc,
                            site_type, mob_rate_per_type, dur_mean_per_type, delta):
    """Simulate a mobility trace for one synthetic individual on a 2D grid (jit for speed)"""
    # Holds tuples of (time_start, time_end, indiv, site, duration)
    data = list()
    # Set rates and probs
    tot_mob_rate = np.sum(mob_rate_per_type)  # Total mobility rate
    site_type_prob = mob_rate_per_type / tot_mob_rate  # Site type probability
    # time
    t = rd.expovariate(tot_mob_rate)
    # Site proximity to individual's home
    site_dist = np.sum((home_loc[indiv] - site_loc)**2,axis=1)
    site_prox = 1/(1+site_dist)

    id = 0
    while t < max_time:
        # Choose a site type
        k = np.searchsorted(np.cumsum(site_type_prob), np.random.random(), side="right")
        s_args = np.where(site_type == k)[0]
        if len(s_args) == 0:  # If there is no site of this type, resample
            # FIXME: If input site types are messed up (prob 1 for missing type)
            # then we end up in an infinit loop...
            continue
        # Choose site: Proportional to distance among chosen type
        site_prob = site_prox[s_args] / site_prox[s_args].sum()
        s_idx = np.random.multinomial(1, pvals=site_prob).argmax()
        site = s_args[s_idx]
        # Duration: Exponential
        dur = rd.expovariate(1/dur_mean_per_type[k])
        if t + dur > max_time:
            break
        # Add visit namedtuple to list
        data.append(Visit(
            id=id,
            t_from=t,
            t_to_shifted=t + dur + delta,
            t_to=t + dur,
            indiv=indiv,
            site=site,
            duration=dur))
        # Shift time to after visit influence (i.e. duration + delta)
        t += dur + delta
        # Shift time to next start of next visit
        t += rd.expovariate(tot_mob_rate)
        # Increment id
        id += 1

    return data

@numba.njit
def _simulate_individual_real_trace(indiv, max_time, site_type, mob_rate_per_type, dur_mean_per_type,
                               variety_per_type, delta, site_dist):
    """Simulate a mobility trace for one real individual in a given town (jit for speed)"""
    # Holds tuples of (time_start, time_end, indiv, site, duration)
    data = list()
    # Set rates and probs
    tot_mob_rate = np.sum(mob_rate_per_type)  # Total mobility rate
    site_type_prob = mob_rate_per_type / tot_mob_rate  # Site type probability
    # time
    t = rd.expovariate(tot_mob_rate)
    # Site proximity to individual's home
    site_dist = site_dist**2
    site_prox = 1/(1+site_dist)

    # Choose usual sites: Inversely proportional to squared distance among chosen type
    usual_sites=[]
    for k in range(len(mob_rate_per_type)):
        usual_sites_k=[]
        # All sites of type k
        s_args = np.where(site_type == k)[0]

        # Number of discrete sites to choose from type k
        variety_k = variety_per_type[k]
        # Probability of sites of type k
        site_prob = site_prox[s_args] / site_prox[s_args].sum()
        done = 0
        while (done < variety_k and len(s_args) > done):
            # s_idx = np.random.choice(site_prob.shape[0], p=site_prob)
            # s_idx = np.random.multinomial(1, pvals=site_prob).argmax()

            # numba-stable/compatible way of np.random.choice (otherwise crashes)
            s_idx = np.searchsorted(np.cumsum(site_prob), np.random.random(), side="right")
            site = s_args[s_idx]
            # Don't pick the same site twice
            if site not in usual_sites_k:
                usual_sites_k.append(site)
                done+=1

        usual_sites.append(usual_sites_k)

    id = 0
    while t < max_time:
        # k = np.random.multinomial(1, pvals=site_type_prob).argmax()
        # k = np.random.choice(site_type_prob.shape[0], p=site_type_prob)

        # Choose a site type
        # numba-stable/compatible way of np.random.choice (otherwise crashes)
        k = np.searchsorted(np.cumsum(site_type_prob), np.random.random(), side="right")

        # Choose a site among the usuals of type k
        site = np.random.choice(np.array(usual_sites[k]))

        # Duration: Exponential
        dur = rd.expovariate(1/dur_mean_per_type[k])
        if t + dur > max_time:
            break
        # Add visit namedtuple to list
        data.append(Visit(
            id=id,
            t_from=t,
            t_to_shifted=t + dur + delta,
            t_to=t + dur,
            indiv=indiv,
            site=site,
            duration=dur))
        # Shift time to after visit influence (i.e. duration + delta)
        t += dur + delta
        # Shift time to next start of next visit
        t += rd.expovariate(tot_mob_rate)
        # Increment id
        id += 1

    return data

@numba.njit
def _simulate_synthetic_mobility_traces(*, num_people, num_sites, max_time, home_loc, site_loc,
                            site_type, people_age, mob_rate_per_age_per_type, dur_mean_per_type,
                            delta, seed):
    rd.seed(seed)
    np.random.seed(seed-1)
    data, visit_counts = list(), list()

    for i in range(num_people):

        # use mobility rates of specific age group
        mob_rate_per_type = mob_rate_per_age_per_type[people_age[i]]
        data_i = _simulate_individual_synthetic_trace(
            indiv=i,
            num_sites=num_sites,
            max_time=max_time,
            home_loc=home_loc,
            site_loc=site_loc,
            site_type=site_type,
            mob_rate_per_type=mob_rate_per_type,
            dur_mean_per_type=dur_mean_per_type,
            delta=delta)

        data.extend(data_i)
        visit_counts.append(len(data_i))

    return data, visit_counts

@numba.njit
def _simulate_real_mobility_traces(*, num_people, max_time, site_type, people_age, mob_rate_per_age_per_type,
                            dur_mean_per_type, home_tile, tile_site_dist, variety_per_type, delta, seed):
    rd.seed(seed)
    np.random.seed(seed-1)
    data, visit_counts = list(), list()

    for i in range(num_people):
        # use mobility rates of specific age group
        mob_rate_per_type = mob_rate_per_age_per_type[people_age[i]]
        # use site distances from specific tiles
        site_dist = tile_site_dist[home_tile[i]]

        data_i = _simulate_individual_real_trace(
            indiv=i,
            max_time=max_time,
            site_type=site_type,
            mob_rate_per_type=mob_rate_per_type,
            dur_mean_per_type=dur_mean_per_type,
            delta=delta,
            variety_per_type=variety_per_type,
            site_dist=site_dist)

        data.extend(data_i)
        visit_counts.append(len(data_i))

    return data, visit_counts


class MobilitySimulator:
    """Simulate a random history of contacts between individuals as follows:
    - Locations of individuals' homes and locations of sites are sampled
    uniformly at random location on a 2D square grid or given as input.
    - Each individual visits a site with rate `1/mob_mean` and remains
    there for `1/duration_mean` (+ a `fixed` delta delay) where mob_mean and
    duration_mean depend on the type of site and the age group of the individual.
    - Individuals choose a site inversely proportional to its distance from their home.
    - Contacts are directional. We define contact from `i` to `j` when:
        - either individuals `i` and `j` were at the same site at the same time,
        - or individual `i` arrived within a `delta` time after `j` left

    The times are reported in the same units, the parameters are given.

    Example of usage to simulate a history for 10 peoples accross 5 sites for
    an observation window of 24 time units:
    ```
    sim = mobilitysim.MobilitySimulator(num_people=10, num_sites=5)
    contacts = sim.simulate(max_time=24)
    ```

    To find if an individual `i` is at site `k` at time `t`, do:
    ```
    sim.is_individual_at_site(indiv=i, site=k, t=t)
    ```

    To find if an individual `i` is contact with individual `j` at site `k`
    at time `t`, do:
    ```
    sim.is_in_contact(indiv_i=i, indiv_j=j, site=k, t=t)
    ```

    To find if an individual `i` will ever be in contact with individual `j` at
    site `k` at any time larger or equal to `t`, do:
    ```
    sim.will_be_in_contact(indiv_i=i, indiv_j=j, site=k, t=t)
    ```

    To find the next contact time with individual `i` with individual `j` at
    site `k`after time `t`, do:
    ```
    sim.next_contact_time(indiv_i=i, indiv_j=j, site=k, t=t)
    ```
    """

    def __init__(self, delta, home_loc=None, people_age=None, site_loc=None, site_type=None,
                site_dict=None, daily_tests_unscaled=None, region_population=None,
                mob_rate_per_age_per_type=None, dur_mean_per_type=None, home_tile=None,
                tile_site_dist=None, variety_per_type=None, people_household=None, downsample=None,
                num_people=None, num_people_unscaled=None, num_sites=None, mob_rate_per_type=None,
                dur_mean=None, num_age_groups=None, seed=None, beacon_config=None, verbose=False):
        """
        delta : float
            Time delta to extend contacts
        home_loc : list of [float,float]
            Home coordinates of each individual
        people_age : list of int
            Age group of each individual
        people_household : list of int
            Household of each individual
        households : dict with key=household, value=individual
            Individuals on each household
        site_loc : list of [float,float]
            Site coordinates
        site_type : list of int
            Type of each site
        site_dict : dict of str
            Translates numerical site types into words
        daily_tests_unscaled : int
            Daily testing capacity per 100k people
        region_population : int
            Number of people living in entire area/region
        downsample : int
            Downsampling factor chosen for real town population and sites
        mob_rate_per_age_per_type: list of list of float
            Mean number of visits per time unit.
            Rows correspond to age groups, columns correspond to site types.
        dur_mean_per_type : float
            Mean duration of a visit per site type
        home_tile : list of int
            Tile indicator for each home
        tile_site_dist: 2D int array
            Pairwise distances between tile centers and sites.
            Rows correspond to tiles, columns correspond to sites.
        variety_per_type : list of int
            Number of discrete sites per type
        num_people : int
            Number of people to simulate
        num_people_unscaled : int
            Real number of people in town (unscaled)
        num_sites : int
            Number of sites to simulate
        mob_rate_per_type : list of floats
            Mean rate for each type of site, i.e. number of visits per time unit
        dur_mean : float
            Mean duration of a visit
        num_age_groups : int
            Number of age groups
        beacon_config: dict
            Beacons implementation configuration
        verbose : bool (optional, default: False)
            Verbosity level
        """

        # Set random seed for reproducibility
        seed = seed or rd.randint(0, 2**32 - 1)
        rd.seed(seed)
        np.random.seed(seed-1)
        
        synthetic = (num_people is not None and num_sites is not None and mob_rate_per_type is not None and
                    dur_mean is not None and num_age_groups is not None)

        real = (home_loc is not None and people_age is not None and site_loc is not None and site_type is not None and
                daily_tests_unscaled is not None and num_people_unscaled is not None and region_population is not None and
                mob_rate_per_age_per_type is not None and dur_mean_per_type is not None and home_tile is not None and
                tile_site_dist is not None and variety_per_type is not None and downsample is not None)

        assert (synthetic != real), 'Unable to decide on real or synthetic mobility generation based on given arguments'

        if synthetic:

            self.mode = 'synthetic'

            self.region_population = None
            self.downsample = None
            self.num_people = num_people
            self.num_people_unscaled = None
            # Random geographical assignment of people's home on 2D grid
            self.home_loc = np.random.uniform(0.0, 1.0, size=(self.num_people, 2))
            
            # Age-group of individuals
            self.people_age = np.random.randint(low=0, high=num_age_groups,
                                                size=self.num_people, dtype=int)
            self.people_household = None
            self.households = None
            self.daily_tests_unscaled =None

            self.num_sites = num_sites
            # Random geographical assignment of sites on 2D grid
            self.site_loc = np.random.uniform(0.0, 1.0, size=(self.num_sites, 2))
            
            # common mobility rate for all age groups
            self.mob_rate_per_age_per_type = np.tile(mob_rate_per_type,(num_age_groups,1))
            self.num_age_groups = num_age_groups
            self.num_site_types = len(mob_rate_per_type)
            # common duration for all types
            self.dur_mean_per_type = np.array(self.num_site_types*[dur_mean])
            
            # Random type for each site
            site_type_prob = np.ones(self.num_site_types)/self.num_site_types
            self.site_type = np.random.multinomial(
                n=1, pvals=site_type_prob, size=self.num_sites).argmax(axis=1)
            
            self.variety_per_type = None
            
            self.home_tile=None
            self.tile_site_dist=None

        elif real:

            self.mode = 'real'

            self.downsample = downsample
            self.region_population = region_population
            self.num_people_unscaled = num_people_unscaled
            self.num_people = len(home_loc)
            self.home_loc = np.array(home_loc)

            self.people_age = np.array(people_age)
            
            if people_household is not None:
                self.people_household = np.array(people_household)
            
                # create dict of households, to retreive household members in O(1) during household infections
                self.households = {}
                for i in range(self.num_people):
                    if self.people_household[i] in self.households:
                        self.households[people_household[i]].append(i)
                    else:
                        self.households[people_household[i]] = [i]
            else:
                self.people_household = None
                self.households = {}

            self.num_sites = len(site_loc)
            self.site_loc = np.array(site_loc)

            self.daily_tests_unscaled = daily_tests_unscaled

            self.mob_rate_per_age_per_type = np.array(mob_rate_per_age_per_type)
            self.num_age_groups = self.mob_rate_per_age_per_type.shape[0]
            self.num_site_types = self.mob_rate_per_age_per_type.shape[1]
            self.dur_mean_per_type = np.array(dur_mean_per_type)
            
            self.site_type = np.array(site_type)

            self.variety_per_type=np.array(variety_per_type)

            self.home_tile=np.array(home_tile)
            self.tile_site_dist=np.array(tile_site_dist)

        else:
            raise ValueError('Provide more information for the generation of mobility data.')

        # Only relevant if an old settings file is being used, should be removed in the future
        if site_dict is None:
            self.site_dict = {0: 'education', 1: 'social', 2: 'bus_stop', 3: 'office', 4: 'supermarket'}
        else:
            self.site_dict = site_dict
        self.delta = delta
        self.verbose = verbose

        self.beacon_config = beacon_config
        self.site_has_beacon = self.place_beacons(
            beacon_config=beacon_config, rollouts=3, max_time=60)

    '''Beacon information computed at test time'''

    def compute_site_priority_by_frequency(self, rollouts, max_time):
        time_at_site = np.zeros(self.num_sites)
        for _ in range(rollouts):
            self.simulate(max_time=max_time, lazy_contacts=True)
            for v in self.all_mob_traces:
                time_at_site[v.site] += v.duration
        temp = time_at_site.argsort()
        site_priority = np.empty_like(temp)
        site_priority[temp] = np.arange(len(time_at_site))
        return site_priority

    def place_beacons(self, *, beacon_config, rollouts, max_time):
        '''
        Computes whether or not a given site has a beacon installed
        '''

        if beacon_config is None:
            return np.zeros(self.num_sites, dtype=bool)
        
        elif beacon_config['mode'] == 'all':
            return np.ones(self.num_sites, dtype=bool)
        
        elif beacon_config['mode'] == 'random':
            # extract mode specific information
            p = beacon_config['p']
            
            # compute beacon locations
            return np.random.binomial(1, p, size=self.num_sites).astype(np.bool)
            
        elif beacon_config['mode'] == 'visit_freq':
            # extract mode specific information
            proportion_with_beacon = beacon_config['proportion_with_beacon']
            
            # compute beacon locations
            site_has_beacon = np.zeros(self.num_sites, dtype=bool)
            site_priority = self.compute_site_priority_by_frequency(rollouts, max_time)
            for k in range(len(site_has_beacon)):
                if site_priority[k] > max(site_priority) * (1 - proportion_with_beacon):
                    site_has_beacon[k] = True
            return site_has_beacon
        
        else:
            raise ValueError('Invalid `beacon_config` mode.')

    '''Class methods'''

    @staticmethod
    def from_pickle(path):
        """
        Load object from pickle file located at `path`

        Parameters
        ----------
        path : str
            Path to input file

        Return
        ------
        sim : MobilitySimulator
            The loaded object
        """
        with open(path, 'rb') as fp:
            obj = pickle.load(fp)
        return obj

    def to_pickle(self, path):
        """
        Save object to pickle file located at `path`

        Parameters
        ----------
        path : str
            Path to output file
        """
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)

    def _simulate_mobility(self, max_time, seed=None):
        """
        Simulate mobility of all people for `max_time` time units

        Parameters
        ----------
        max_time : float
            Number time to simulate
        seed : int
            Random seed for reproducibility

        Return
        ------
        mob_traces : list of `Visit` namedtuples
            List of simulated visits of individuals to sites
        home_loc : numpy.ndarray
            Locations of homes of individuals
        site_loc : numpy.ndarray
            Locations of sites
        """
        # Set random seed for reproducibility
        seed = seed or rd.randint(0, 2**32 - 1)
        rd.seed(seed)
        np.random.seed(seed-1)

        if self.mode == 'synthetic':
            all_mob_traces, self.visit_counts = _simulate_synthetic_mobility_traces(
                num_people=self.num_people,
                num_sites=self.num_sites,
                max_time=max_time,
                home_loc=self.home_loc,
                site_loc=self.site_loc,
                site_type=self.site_type,
                people_age=self.people_age,
                mob_rate_per_age_per_type=self.mob_rate_per_age_per_type,
                dur_mean_per_type=self.dur_mean_per_type,
                delta=self.delta,
                seed=rd.randint(0, 2**32 - 1)
                )

        elif self.mode == 'real':
            all_mob_traces, self.visit_counts = _simulate_real_mobility_traces(
                num_people=self.num_people,
                max_time=max_time,
                site_type=self.site_type,
                people_age=self.people_age,
                mob_rate_per_age_per_type=self.mob_rate_per_age_per_type,
                dur_mean_per_type=self.dur_mean_per_type,
                delta=self.delta,
                home_tile=self.home_tile,
                variety_per_type=self.variety_per_type,
                tile_site_dist=self.tile_site_dist,
                seed=rd.randint(0, 2**32 - 1)
                )

        # Group mobility traces per indiv 
        self.mob_traces = self._group_mob_traces(all_mob_traces)
        return all_mob_traces

    def _find_contacts(self):
        """
        Finds contacts in a given list `mob_traces` of `Visit`s
        and stores them in a dictionary of dictionaries of InterLap objects,
        """
        # Group mobility traces by site
        mob_traces_at_site = defaultdict(list)
        for v in self.all_mob_traces:
            mob_traces_at_site[v.site].append(v)

        contacts = self._find_mob_trace_overlaps(sites=range(self.num_sites),
                                                 mob_traces_at_site=mob_traces_at_site,
                                                 infector_mob_traces_at_site=mob_traces_at_site,
                                                 tmin=0.0,
                                                 for_all_individuals=True)
        return contacts

    def find_contacts_of_indiv(self, indiv, tmin, tmax):
        """
        Finds all delta-contacts of person 'indiv' with any other individual after time 'tmin'
        and returns them as InterLap object.
        In the simulator, this function is called for `indiv` as infector.
        """
        mob_traces_at_site = defaultdict(list)
        infector_mob_traces_at_site = defaultdict(list)
        visited_sites = []

        for v in self.all_mob_traces:
            # Only consider visit if overlap with [tmin, tmax] 
            if tmin < v.t_to_shifted and (v.t_from < tmax if (tmax is not None) else True):

                # Store as either mob_trace of infector or of everyone else
                if v.indiv == indiv:
                    infector_mob_traces_at_site[v.site].append(v)
                    if v.site not in visited_sites:
                        visited_sites.append(v.site)
                mob_traces_at_site[v.site].append(v)

        contacts = self._find_mob_trace_overlaps(sites=visited_sites,
                                                 mob_traces_at_site=mob_traces_at_site,
                                                 infector_mob_traces_at_site=infector_mob_traces_at_site,
                                                 tmin=tmin,
                                                 tmax=tmax,
                                                 for_all_individuals=False)
        return contacts

    def _find_mob_trace_overlaps(self, sites, mob_traces_at_site, infector_mob_traces_at_site, tmin, tmax, for_all_individuals):

        # decide way of storing depending on way the function is used (all or individual)
        # FIXME: this could be done in a cleaner way by calling this function several times in `_find_contacts` 
        if for_all_individuals:
            # dict of dict of list of contacts:
            # i.e. contacts[i][j][k] = "k-th contact from i to j"
            contacts = {i: defaultdict(InterLap) for i in range(self.num_people)}
        else:
            contacts = InterLap()

        if self.verbose and for_all_individuals:
            print() # otherwise jupyter notebook looks ugly

        for s in sites:
            if self.verbose and for_all_individuals:
                print('Checking site ' + str(s + 1) + '/' + str(len(sites)), end='\r')
            if len(mob_traces_at_site[s]) == 0:
                continue

            # Init the interval overlap matcher
            inter = InterLap()
            inter.update(mob_traces_at_site[s])

            # Match contacts
            # Iterate over each visit of the infector at site s
            for v_inf in infector_mob_traces_at_site[s]:

                # Skip if delta-contact ends before `tmin` or begins after `tmax`
                if tmin < v_inf.t_to_shifted and (v_inf.t_from < tmax if (tmax is not None) else True):
                    
                    v_time = (v_inf.t_from, v_inf.t_to_shifted)

                    # Find any othe person that had overlap with this visit 
                    for v in list(inter.find(other=v_time)):

                        # Ignore contacts with same individual
                        if v.indiv == v_inf.indiv:
                            continue

                        # Compute contact time
                        c_t_from = max(v.t_from, v_inf.t_from)
                        c_t_to = min(v.t_to, v_inf.t_to_shifted)
                        if c_t_to > c_t_from and c_t_to > tmin:

                            # Init contact tuple
                            # Note 1: Contact always considers delta overlap for `indiv_j` 
                            # (i.e. for `indiv_j` being the infector)
                            # Note 2: Contact contains the delta-extended visit of `indiv_j`
                            # (i.e. there is a `Contact` even when `indiv_j` never overlapped physically with `indiv_i`)
                            # (i.e. need to adjust for that in dY_i integral)
                            c = Contact(t_from=c_t_from,
                                        t_to=c_t_to,
                                        indiv_i=v.indiv,
                                        indiv_j=v_inf.indiv,
                                        id_tup=(v.id, v_inf.id),
                                        site=s,
                                        duration=c_t_to - c_t_from)

                            # Add it to interlap
                            if for_all_individuals:
                                # Dictionary of all contacts
                                contacts[v.indiv][v_inf.indiv].update([c])
                            else:
                                # All contacts of (infector) 'indiv' only
                                contacts.update([c])
        return contacts

    def _group_mob_traces(self, mob_traces):
        """Group `mob_traces` by individual and for faster queries.
        Returns a dict of dict of Interlap of the form:

            mob_traces_dict[i] = "Interlap of visits of indiv i"
        """
        mob_traces_dict = {i: InterLap() for i in range(self.num_people)}
        for v in mob_traces:
            mob_traces_dict[v.indiv].update([v])
        return mob_traces_dict

    def simulate(self, max_time, seed=None, lazy_contacts=False):
        """
        Simulate contacts between individuals in time window [0, max_time].

        Parameters
        ----------
        max_time : float
            Maximum time to simulate
        seed : int
            Random seed for mobility simulation
        lazy_contacts : bool
            If true the contact dictionary is not computed and contacts
            need to be computed on-the-fly during launch_epidemic

        Returns
        -------
        contacts : list of list of tuples
            A list of namedtuples containing the list of all contacts as
            namedtuples ('time_start', 'indiv_j', 'duration'), where:
            - `time_start` is the time the contact started
            - 'indiv_j' is the id of the individual the contact was with
            - 'duration' is the duration of the contact
        """
        self.max_time = max_time

        # Simulate mobility of each individuals to each sites
        if self.verbose:
            print(f'Simulate mobility for {max_time:.2f} time units... ',
                  end='', flush=True)
        all_mob_traces = self._simulate_mobility(max_time, seed)
        self.all_mob_traces = all_mob_traces

        if self.verbose:
            print(f'Simulated {len(all_mob_traces)} visits.', flush=True)

        if not lazy_contacts:
            # Find the contacts in all sites in the histories
            if self.verbose:
                print(f'Find contacts... ', end='')
            self.contacts = self._find_contacts()

        else:
            # Initialize empty contact array
            self.contacts = {i: defaultdict(InterLap) for i in range(self.num_people)}


    def list_intervals_in_window_individual_at_site(self, *, indiv, site, t0, t1):
        """Return a generator of Intervals of all visits of `indiv` is at site
           `site` that overlap with [t0, t1]
        """
        for visit in self.mob_traces[indiv].find((t0, t1)):
            # the above call matches all on (`t_from`, `t_to_shifted`)
            # thus need to filter out visits that ended before `t0`, 
            # i.e. visits such that `t_to` <= `t0`, 
            # i.e. environmental match only (match only occured on (`t_to`, `t_to_shifted`))

            if visit.t_to > t0 and visit.site == site:
                yield Interval(visit.t_from, visit.t_to)

    def is_in_contact(self, *, indiv_i, indiv_j, t, site=None):
        """Indicate if individuals `indiv_i` is within `delta` time to
        make contact with `indiv_j` at time `t` in site `site`, and return contact if possible
        """
        try:
            # Find contact matching time and check site
            contact = next(self.contacts[indiv_i][indiv_j].find((t, t)))
            return (site is None) or (contact.site == site), contact

        except StopIteration:  # No such contact, call to `next` failed
            return False, None

    def will_be_in_contact(self, *, indiv_i, indiv_j, t, site=None):
        """Indicate if individuals `indiv_i` will ever make contact with
        `indiv_j` in site `site` at a time greater or equal to `t`
        """
        contacts_ij = self.contacts[indiv_i][indiv_j]
        # Search future contacts
        for c in contacts_ij.find((t, np.inf)):
            # Check site
            if site is None:
                return True
            elif (site is None) or (c.site == site):
                return True

        return False

    def next_contact(self, *, indiv_i, indiv_j, t=np.inf, site=None):
        """Returns the next `delta`- contact between
            `indiv_i` with `indiv_j` in site `site` at a time greater or equal to `t`
        """
        contacts_ij = self.contacts[indiv_i][indiv_j]
        # Search future contacts
        for c in contacts_ij.find((t, np.inf)):
            # Check site
            if (site is None) or (c.site == site):
                return c
        return None # No contact in the future
