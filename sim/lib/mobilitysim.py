from collections import namedtuple, defaultdict
import itertools
import random as rd
import pandas as pd
import numpy as np
import numba
import pickle
import json

from interlap import InterLap


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
                mob_rate_per_age_per_type=None, dur_mean_per_type=None, home_tile=None,
                tile_site_dist=None, variety_per_type=None,
                num_people=None, num_sites=None, mob_rate_per_type=None, dur_mean=None,
                num_age_groups=None, verbose=False):
        """
        delta : float
            Time delta to extend contacts
        home_loc : list of [float,float]
            Home coordinates of each individual
        people_age : list of int
            Age group of each individual
        site_loc : list of [float,float]
            Site coordinates
        site_type : list of int
            Type of each site
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
        num_sites : int
            Number of sites to simulate
        mob_rate_per_type : list of floats
            Mean rate for each type of site, i.e. number of visits per time unit
        dur_mean : float
            Mean duration of a visit
        num_age_groups : int
            Number of age groups
        verbose : bool (optional, default: False)
            Verbosity level
        """

        synthetic = (num_people is not None and num_sites is not None and mob_rate_per_type is not None and
                    dur_mean is not None and num_age_groups is not None)

        real = (home_loc is not None and people_age is not None and site_loc is not None and site_type is not None and
                mob_rate_per_age_per_type is not None and dur_mean_per_type is not None and home_tile is not None and
                tile_site_dist is not None and variety_per_type is not None)
        
        assert (synthetic != real), 'Unable to decide on real or synthetic mobility generation based on given arguments'

        if synthetic:
            
            self.mode = 'synthetic'
            
            self.num_people = num_people
            self.num_sites = num_sites

            self.num_site_types = len(mob_rate_per_type)
            self.num_age_groups = num_age_groups
            
            # common duration for all types
            self.dur_mean_per_type = np.array(self.num_site_types*[dur_mean])
            # common mobility rate for all age groups
            self.mob_rate_per_age_per_type = np.tile(mob_rate_per_type,(num_age_groups,1))

            self.home_tile=None
            self.tile_site_dist=None
            self.variety_per_type=None

        elif real:

            self.mode = 'real'

            self.num_people = len(home_loc)
            self.home_loc = np.array(home_loc)

            self.people_age = np.array(people_age)

            self.num_sites = len(site_loc)
            self.site_loc = np.array(site_loc)

            self.site_type = np.array(site_type)

            self.mob_rate_per_age_per_type = np.array(mob_rate_per_age_per_type)
            self.num_age_groups = self.mob_rate_per_age_per_type.shape[0]
            self.num_site_types = self.mob_rate_per_age_per_type.shape[1]
            self.dur_mean_per_type = np.array(dur_mean_per_type)

            self.variety_per_type=np.array(variety_per_type)

            self.home_tile=np.array(home_tile)
            self.tile_site_dist=np.array(tile_site_dist)

        else:
            raise ValueError('Provide more information for the generation of mobility data.')
                
        self.delta = delta
        self.verbose = verbose

    @staticmethod
    def from_json(fp, compute_contacts=True):
        """
        Reach the from `fp` (.read()-supporting file-like object) that is
        expected to be JSON-formated from the `to_json` file.

        Parameters
        ----------
        fp : object
            The input .read()-supporting file-like object
        compute_contacts : bool (optional, default: True)
            Indicate if contacts should be computed from the mobility traces.
            If True, then any `contact` key in `fp` will be ignored.
            If False, `fp` must have a contact` key.

        Return
        ------
        sim : MobilitySimulator
            The loaded object
        """
        # Read file into json dict
        data = json.loads(fp.read())

        # Init object
        init_attrs = ['num_people', 'num_sites', 'delta',
                      'mob_mean', 'dur_mean', 'verbose']
        obj = MobilitySimulator(**{attr: data[attr] for attr in init_attrs})

        # Set np.ndarray attributes
        for attr in ['home_loc', 'site_loc']:
            setattr(obj, attr, np.array(data[attr]))

        # Set list attributes
        for attr in ['visit_counts']:
            setattr(obj, attr, list(data[attr]))

        # Set `mob_traces` attribute into dict:defaultdict:InterLap
        setattr(obj, 'mob_traces', {i: defaultdict(InterLap) for i in range(obj.num_people)})
        for indiv, traces_i in data['mob_traces'].items():
            indiv = int(indiv)  # JSON does not support int keys
            for site, visit_list in traces_i.items():
                site = int(site)  # JSON does not support int keys
                if len(visit_list) > 0:
                    inter = InterLap()
                    inter.update(list(map(lambda t: Visit(*t), visit_list)))
                    obj.mob_traces[indiv][site] = inter

        # Set `contacts` attribute into dict:defaultdict:InterLap
        if compute_contacts:  # Compute from `mob_traces`
            all_mob_traces = []
            for i, traces_i in obj.mob_traces.items():
                for j, inter in traces_i.items():
                    all_mob_traces.extend(inter._iset)
            # Compute contacts from mobility traces
            obj.contacts = obj._find_contacts(all_mob_traces)
        else:  # Load from file
            setattr(obj, 'contacts', {i: defaultdict(InterLap) for i in range(obj.num_people)})
            for indiv_i, contacts_i in data['contacts'].items():
                indiv_i = int(indiv_i)  # JSON does not support int keys
                for indiv_j, contact_list in contacts_i.items():
                    indiv_j = int(indiv_j)  # JSON does not support int keys
                    if len(contact_list) > 0:
                        inter = InterLap()
                        inter.update(list(map(lambda t: Contact(*t), contact_list)))
                        obj.contacts[indiv_i][indiv_j] = inter

        return obj

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
            # Random geographical assignment of people's home on 2D grid
            self.home_loc = np.random.uniform(0.0, 1.0, size=(self.num_people, 2))
            # Age-group of individuals
            self.people_age = np.random.randint(low=0, high=self.num_age_groups,
                                        size=self.num_people, dtype=int)
            # Random geographical assignment of sites on 2D grid
            self.site_loc = np.random.uniform(0.0, 1.0, size=(self.num_sites, 2))
            # Random type for each site
            site_type_prob = np.ones(self.num_site_types)/self.num_site_types
            self.site_type = np.random.multinomial(
                n=1, pvals=site_type_prob, size=self.num_sites).argmax(axis=1)

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

        # Group mobility traces per indiv and site
        self.mob_traces = self._group_mob_traces(all_mob_traces)
        return all_mob_traces

    def _find_contacts(self, mob_traces):
        """Find contacts in a given list `mob_traces` of `Visit`s"""
        # Group mobility traces by site
        mob_traces_at_site = defaultdict(list)
        for v in mob_traces:
            mob_traces_at_site[v.site].append(v)

        # dict of dict of list of contacts:
        # i.e. contacts[i][j][k] = "k-th contact from i to j"
        contacts = {i: defaultdict(InterLap) for i in range(self.num_people)}

        # For each site s
        for s in range(self.num_sites):
            if self.verbose:
                print('Checking site '+str(s+1)+'/'+str(self.num_sites), end='\r')
            if len(mob_traces_at_site[s]) == 0:
                continue
            
            # Init the interval overlap matcher
            inter = InterLap()
            inter.update(mob_traces_at_site[s])
            # Match contacts
            for v in mob_traces_at_site[s]:
                v_time = (v.t_from, v.t_to)
                for vo in list(inter.find(other=v_time)):
                    # Ignore contacts with same individual
                    if v.indiv == vo.indiv:
                        continue
                    # Compute contact time
                    c_t_from = max(v.t_from, vo.t_from)
                    c_t_to = min(v.t_to, vo.t_to_shifted)
                    if c_t_to > c_t_from:
                        # Set contact tuple
                        c = Contact(t_from=c_t_from,
                                    t_to=c_t_to,
                                    indiv_i=v.indiv,
                                    indiv_j=vo.indiv,
                                    id_tup=(v.id, vo.id),
                                    site=s,
                                    duration=c_t_to - c_t_from)
                        # Add it to interlap
                        contacts[v.indiv][vo.indiv].update([c])

        return contacts

    def _group_mob_traces(self, mob_traces):
        """Group `mob_traces` by individual and site for faster queries.
        Returns a dict of dict of Interlap of the form:

            mob_traces_dict[i][s] = "Interlap of visits of indiv i at site s"
        """
        mob_traces_dict = {i: defaultdict(InterLap) for i in range(self.num_people)}
        for v in mob_traces:
            mob_traces_dict[v.indiv][v.site].update([v])
        return mob_traces_dict

    def simulate(self, max_time, seed=None):
        """
        Simulate contacts between individuals in time window [0, max_time].

        Parameters
        ----------
        max_time : float
            Maximum time to simulate
        seed : int
            Random seed for mobility simulation

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
        if self.verbose:
            print(f'Simulated {len(all_mob_traces)} visits.', flush=True)

        # Find the contacts in all sites in the histories
        if self.verbose:
            print(f'Find contacts... ', end='')
        self.contacts = self._find_contacts(all_mob_traces)
        # FIXME: contact_count calculation takes too long
        # self.contact_count = sum(len(self.contacts[i][j]) for i in range(
        #     self.num_people) for j in range(self.num_people))
        # if self.verbose:
        #     print(f'Found {self.contact_count} contacts', flush=True)

    def is_individual_at_site(self, indiv, site, t):
        """Indicate if individual `indiv` is present at site `site` at time `t`
            and returns interval if possible
        Returns:
            If true: True, Interval
            If False: False, None
        """
        matches = list(self.mob_traces[indiv][site].find((t,t)))
        if len(matches) == 1:
            visit = matches[0]
            # Match on (`t_form`, `t_to_shifted`), need to filter out matches
            # after time `t_to`
            # FIXME: This could be made easier by using the non-shifted
            # intervals in `self.mob_traces`
            if t < visit.t_to:
                return True, Interval(matches[0].t_from, matches[0].t_to)
        elif len(matches) > 1:
            # An indiv cannot be at the site more than once at the same time
            raise RuntimeError(("Too many matches found, that's not possible, "
                                "you probably found a bug..."))
        # No match
        return False, None

    def list_intervals_in_window_individual_at_site(self, indiv, site, t0, t1):
        """Return a generator of Intervals of all visits of `indiv` is at site
           `site` that overlap with [t0, t1]

            FIXME: Make sure that this query is correct
        """
        for visit in self.mob_traces[indiv][site].find((t0, t1)):
            # Match on (`t_form`, `t_to_shifted`), need to filter out visits
            # that ended before `t0`, i.e. visits such that `t_to` <= `t0`
            # FIXME: This could be made easier by using the non-shifted
            # intervals in `self.mob_traces`
            if visit.t_to > t0:
                yield Interval(visit.t_from, visit.t_to)

    def is_in_contact(self, indiv_i, indiv_j, site, t):
        """Indicate if individuals `indiv_i` is within `delta` time to
        make contact with `indiv_j` at time `t` in site `site`, and return contact if possible
        """
        try:
            # Find contact matching time and check site
            contact = next(self.contacts[indiv_i][indiv_j].find((t, t)))
            return contact.site == site, contact

        except StopIteration:  # No such contact, call to `next` failed
            return False, None

    def will_be_in_contact(self, indiv_i, indiv_j, site, t):
        """Indicate if individuals `indiv_i` will ever make contact with
        `indiv_j` in site `site` at a time greater or equal to `t`
        """
        contacts_ij = self.contacts[indiv_i][indiv_j]
        # Search future contacts
        for c in contacts_ij.find((t, np.inf)):
            # Check site
            if site is None:
                return True
            elif c.site == site:
                return True
        
        return False
    
    def next_contact(self, indiv_i, indiv_j, site=None, t=np.inf):
        """Returns the next `delta`- contact between 
            `indiv_i` with `indiv_j` in site `site` at a time greater or equal to `t`
        """
        contacts_ij = self.contacts[indiv_i][indiv_j]
        # Search future contacts
        for c in contacts_ij.find((t, np.inf)):
            # Check site
            if c.site == site:
                return c
        return None # No contact in the future

