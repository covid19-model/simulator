import abc
from collections import namedtuple, defaultdict
import numpy as np
import math

from interlap import InterLap

# from utils import enforce_init_run # use this for unit testing
from lib.utils import enforce_init_run


# Tuple representing an interval, FIXME: duplicates mobilitysim Interval
Interval = namedtuple('Interval', ('left', 'right'))

# Small time subtracted from the end of time windows to avoid matching at
# limit between two measures, because interlap works with closed intervals
EPS = 1e-15


class Measure(metaclass=abc.ABCMeta):

    def __init__(self, t_window):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        """
        if not isinstance(t_window, Interval):
            raise ValueError('`t_window` must be an Interval namedtuple')
        self.t_window = t_window
        # Set init run attribute
        self._is_init = False

    def init_run(self, **kwargs):
        """Init the measure for this run with whatever is needed"""
        raise NotImplementedError(("Must be implemented in child class. If you"
                                   " get this error, it's probably a bug."))

    def _in_window(self, t):
        """Indicate if the measure is valid, i.e. if time `t` is in the time
        window of the measure"""
        return (t >= self.t_window.left) and (t < self.t_window.right)

"""
=========================== SOCIAL DISTANCING ===========================
"""


class SocialDistancingForAllMeasure(Measure):
    """
    Social distancing measure. All the population is advised to stay home. Each
    visit of each individual respects the measure with some probability.
    """

    def __init__(self, t_window, p_stay_home):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        p_stay_home : float
            Probability of respecting the measure, should be in [0,1]
        """
        # Init time window
        super().__init__(t_window)

        # Init probability of respecting measure
        if (not isinstance(p_stay_home, float)) or (p_stay_home < 0):
            raise ValueError("`p_stay_home` should be a non-negative float")
        self.p_stay_home = p_stay_home

    def init_run(self, n_people, n_visits):
        """Init the measure for this run by sampling the outcome of each visit
        for each individual 

        Parameters
        ----------
        n_people : int
            Number of people in the population
        n_visits : int
            Maximum number of visits of an individual
        """
        # Sample the outcome of the measure for each visit of each individual
        self.bernoulli_stay_home = np.random.binomial(
            1, self.p_stay_home, size=(n_people, n_visits))
        self._is_init = True

    @enforce_init_run
    def is_contained(self, *, j, j_visit_id, t):
        """Indicate if individual `j` respects measure for visit `j_visit_id`
        """
        is_home_now = self.bernoulli_stay_home[j, j_visit_id]
        return is_home_now and self._in_window(t)
    
    @enforce_init_run
    def is_contained_prob(self, *, j, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        if self._in_window(t):
            return self.p_stay_home
        return 0.0


class UpperBoundCasesSocialDistancing(SocialDistancingForAllMeasure):

    def __init__(self, t_window, p_stay_home, max_pos_tests_per_week=50, intervention_times=None):
        """
        Additional parameters:
        ----------------------
        max_pos_test_per_week : int
            If the number of positive tests per week exceeds this number the measure becomes active
        intervention_times : list of floats
            List of points in time at which interventions can be changed. If 'None' interventions can be changed at any time
        """

        super().__init__(t_window, p_stay_home)
        self.max_pos_tests_per_week = max_pos_tests_per_week
        self.intervention_history = []
        if intervention_times is not None:
            self.intervention_times = np.asarray(intervention_times)
        else:
            self.intervention_times = None

    def _are_cases_above_threshold(self, t, t_pos_tests):
        # If measures can be changed continuously
        if self.intervention_times is None:
            t_intervention = t
        else:  # If measures can be changed at intervention times
            # Find largest time in intervention_times s.t. t > time
            t_intervention = np.where(t - self.intervention_times > 0, t - self.intervention_times, np.inf).min()

        # Count positive tests in last 7 days from last intervention time
        tmin = t_intervention - 7 * 24
        num_pos_tests = np.sum(np.greater(t_pos_tests, tmin) * np.less(t_pos_tests, t_intervention))
        is_measure_active = num_pos_tests > self.max_pos_tests_per_week
        self.intervention_history.append((t, is_measure_active))
        return is_measure_active

    @enforce_init_run
    def is_contained(self, *, j, j_visit_id, t, t_pos_tests):
        """Indicate if individual `j` respects measure for visit `j_visit_id`
        """
        if not self._in_window(t):
            return False

        is_home_now = self.bernoulli_stay_home[j, j_visit_id]
        return is_home_now and self._are_cases_above_threshold(t, t_pos_tests)

    @enforce_init_run
    def is_contained_prob(self, *, j, t, t_pos_tests):
        """Returns probability of containment for individual `j` at time `t`
        """
        if not self._in_window(t):
            return 0.0

        if self._are_cases_above_threshold(t, t_pos_tests):
            return self.p_stay_home
        return 0.0


class SocialDistancingPerStateMeasure(SocialDistancingForAllMeasure):
    """
    Social distancing measure. Only the population in a given 'state' is advised
    to stay home. Each visit of each individual respects the measure with some
    probability.

    NOTE: This is the same as a SocialDistancingForAllMeasure but `is_contained` query also checks that the state 'state' 
    of individual j is True
    """

    def __init__(self, t_window, p_stay_home, state_label):
        # Init time window
        super().__init__(t_window, p_stay_home)
        
        self.state_label = state_label
        
    @enforce_init_run
    def is_contained(self, *, j, j_visit_id, t, state_dict):
        """Indicate if individual `j` is in state 'state' and respects measure for
        visit `j_visit_id`

        r : int
            Id of realization
        j : int
            Id of individual
        j_visit_id : int
            Id of visit
        t : float
            Query time
        state_dict : dict
            Dict with states of all individuals in `DiseaseModel`
        """
        is_home_now = self.bernoulli_stay_home[j, j_visit_id]
        # only isolate at home while at state `state`
        return is_home_now and state_dict[state_label][j] and self._in_window(t)
    
    @enforce_init_run
    def is_contained_prob(self, *, j, t, state_started_at_dict, state_ended_at_dict):
        """Returns probability of containment for individual `j` at time `t`
        """
        if (self._in_window(t) and t >= state_started_at_dict[state_label][j] and t<=state_ended_at_dict[state_label][j]):
            return self.p_stay_home
        return 0.0


class SocialDistancingForPositiveMeasure(SocialDistancingForAllMeasure):
    """
    Social distancing measure. Only the population of positive cases who are not
    resistant or dead is advised to stay home. Each visit of each individual 
    respects the measure with some probability.

    NOTE: This is the same as a SocialDistancingForAllMeasure but `is_contained` query also checks that the state 'posi' of individual j is True
    """

    @enforce_init_run
    def is_contained(self, *, j, j_visit_id, t, state_posi_started_at, state_posi_ended_at, state_resi_started_at, state_dead_started_at):
        """Indicate if individual `j` is positive and respects measure for
        visit `j_visit_id`

        r : int
            Id of realization
        j : int
            Id of individual
        j_visit_id : int
            Id of visit
        t : float
            Query time
        state_* : array
            Array of indicators, it should be the array of `state` `*` of the `DiseaseModel`
        """

        is_home_now = self.bernoulli_stay_home[j, j_visit_id]

        # only isolate at home while positive and not resistant or dead
        is_posi_now = (
            t >= state_posi_started_at[j] and t < state_posi_ended_at[j] and # positive
            t < state_resi_started_at[j] and t < state_dead_started_at[j] # not resistant or dead
        )

        return is_home_now and is_posi_now and self._in_window(t)
    
    @enforce_init_run
    def is_contained_prob(self, *, j, t, state_posi_started_at, state_posi_ended_at, state_resi_started_at, state_dead_started_at):
        """Returns probability of containment for individual `j` at time `t`
        """
        if (self._in_window(t) and 
            t >= state_posi_started_at[j] and t < state_posi_ended_at[j] and # positive
            t < state_resi_started_at[j] and t < state_dead_started_at[j]): # not resistant or dead
            return self.p_stay_home
        return 0.0


class SocialDistancingForPositiveMeasureHousehold(Measure):
    """
    Social distancing measure. Isolate positive cases from household members. 
    Each individual respects the measure with some probability.
    """

    def __init__(self, t_window, p_isolate):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        p_isolate : float
            Probability of respecting the measure, should be in [0,1]
        """
        # Init time window
        super().__init__(t_window)
        self.p_isolate = p_isolate
        
    def init_run(self):
        """Init the measure for this run is trivial
        """
        self._is_init = True
        
    @enforce_init_run
    def is_contained(self, *, j, t, state_posi_started_at, state_posi_ended_at, state_resi_started_at, state_dead_started_at):
        """Indicate if individual `j` respects measure 
        """
        is_isolated = np.random.binomial(1, self.p_isolate)

        # only isolate at home while positive and not resistant or dead
        is_posi_now = (
            t >= state_posi_started_at[j] and t < state_posi_ended_at[j] and # positive
            t < state_resi_started_at[j] and t < state_dead_started_at[j] # not resistant or dead
        )

        return is_isolated and is_posi_now and self._in_window(t)

    @enforce_init_run
    def is_contained_prob(self, *, j, t, state_posi_started_at, state_posi_ended_at, state_resi_started_at, state_dead_started_at):
        """Returns probability of containment for individual `j` at time `t`
        """
        if (self._in_window(t) and 
            t >= state_posi_started_at[j] and t <= state_posi_ended_at[j] and # positive
            t < state_resi_started_at[j] and t < state_dead_started_at[j]): # not resistant or dead
            return p_isolate
        return 0.0
            
class SocialDistancingByAgeMeasure(Measure):
    """
    Social distancing measure. The population is advised to stay at home based
    on membership in a specific age group. The measure defines the probability
    of staying at home for all age groups in the simulation.
    """

    def __init__(self, t_window, p_stay_home):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        p_stay_home : float
            Probability of respecting the measure, should be in [0,1]
        """
        # Init time window
        super().__init__(t_window)

        # Init probability of respecting measure
        if (not isinstance(p_stay_home, list)) or (any(map(lambda x: x < 0, p_stay_home))):
            raise ValueError("`p_stay_home` should be a list of only non-negative floats")
        self.p_stay_home = p_stay_home

    def init_run(self, num_age_groups, n_visits):
        """Init the measure for this run by sampling the outcome of each visit
        for each individual

        Parameters
        ----------
        num_age_groups : int
            Number of ages groups in the population
        n_visits : int
            Maximum number of visits of an individual
        """
        if len(self.p_stay_home) != num_age_groups:
            raise ValueError("`p_stay_home` list is different in DiseaseModel and MobilitySim")

        # Sample the outcome of the measure for each visit of each individual
        self.bernoulli_stay_home = np.random.binomial(
            1, self.p_stay_home, size=(n_visits, num_age_groups))
        self._is_init = True

    @enforce_init_run
    def is_contained(self, *, age, j_visit_id, t):
        """Indicate if individual of age `age` respects measure for visit `j_visit_id`
        """
        is_home_now = self.bernoulli_stay_home[j_visit_id, age]
        return is_home_now and self._in_window(t)
    
    @enforce_init_run
    def is_contained_prob(self, *, age, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        if self._in_window(t):
            return self.p_stay_home[age]
        return 0.0

class SocialDistancingForSmartTracing(Measure):
    """
    Social distancing measure. Only the population who intersected with positive cases 
    for ``test_smart_duration``. Each visit of each individual respects the measure with 
    some probability.

    NOTE: This is the same as a SocialDistancingForAllMeasure but `is_contained` query also checks that the state 'posi' of individual j is True
    """

    def __init__(self, t_window, p_stay_home, test_smart_duration):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        p_stay_home : float
            Probability of respecting the measure, should be in [0,1]
        """
        # Init time window
        super().__init__(t_window)

        # Init probability of respecting measure
        if (not isinstance(p_stay_home, float)) or (p_stay_home < 0):
            raise ValueError("`p_stay_home` should be a non-negative float")
        self.p_stay_home = p_stay_home
        self.test_smart_duration = test_smart_duration

    def init_run(self, n_people, n_visits):
        """Init the measure for this run by sampling the outcome of each visit
        for each individual

        Parameters
        ----------
        n_people : int
            Number of people in the population
        n_visits : int
            Maximum number of visits of an individual
        """
        # Sample the outcome of the measure for each visit of each individual
        self.bernoulli_stay_home = np.random.binomial(
            1, self.p_stay_home, size=(n_people, n_visits))
        self.time_stay_home = -np.inf * np.ones((n_people), dtype='float')
        self.intervals_stay_home = list()
        self._is_init = True
        
    @enforce_init_run
    def is_contained(self, *, j, j_visit_id, t):
        """Indicate if individual `j` respects measure for visit `j_visit_id`
        """
        is_home_now = self.bernoulli_stay_home[j, j_visit_id] and (t < self.time_stay_home[j])
        return is_home_now and self._in_window(t)
    
    @enforce_init_run
    def start_containment(self, *, j, t):
        self.time_stay_home[j] = t + self.test_smart_duration
        self.intervals_stay_home.append((j, t))
        return
    
    @enforce_init_run
    def is_contained_prob(self, *, j, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        if self._in_window(t):
            for interval in self.intervals_stay_home:
                if interval[0] == j and t >= interval[1] and t <= interval[1] + self.test_smart_duration:
                    return self.p_stay_home
        return 0.0


class SocialDistancingForKGroups(Measure):
    """
    Social distancing measure where the population is based on K groups, here their IDs.
    Each day 1 of K groups is allowed to go outside.
    """

    def __init__(self, t_window, K):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        K : int
            Number of groups having to stay home on different days
        """
        # Init time window
        super().__init__(t_window)
        self.K = K
        
    def init_run(self):
        """Init the measure for this run is trivial
        """
        self._is_init = True
        
    @enforce_init_run
    def is_contained(self, *, j, t):
        """Indicate if individual `j` respects measure 
        """
        day = math.floor(t / 24.0)
        is_home_now = ((j % self.K) != (day % self.K)) 
        return is_home_now and self._in_window(t)

    @enforce_init_run
    def is_contained_prob(self, *, j, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        day = math.floor(t / 24.0)
        is_home_now = ((j % self.K) != (day % self.K))
        if is_home_now and self._in_window(t):
            return 1.0
        return 0.0

"""
=========================== SITE SPECIFIC MEASURES ===========================
"""


class BetaMultiplierMeasure(Measure):

    def __init__(self, t_window, beta_multiplier):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        beta_multiplier : list of floats
            List of multiplicative factor to infection rate at each site
        """

        super().__init__(t_window)
        if (not isinstance(beta_multiplier, dict)
            or (min(beta_multiplier.values()) < 0)):
            raise ValueError(("`beta_multiplier` should be dict of"
                              " non-negative floats"))
        self.beta_multiplier = beta_multiplier

    # def beta_factor(self, *args):
    #     """Initialize general beta_factor function"""
    #     raise NotImplementedError(("Must be implemented in child class. If you"
    #                                " get this error, it's probably a bug."))


class BetaMultiplierMeasureBySite(BetaMultiplierMeasure):

    def beta_factor(self, *, k, t):
        """Returns the multiplicative factor for site `k` at time `t`. The
        factor is one if `t` is not in the active time window of the measure.
        """
        return self.beta_multiplier[k] if self._in_window(t) else 1.0


class BetaMultiplierMeasureByType(BetaMultiplierMeasure):

    def beta_factor(self, *, typ, t):
        """Returns the multiplicative factor for site type `typ` at time `t`. The
        factor is one if `t` is not in the active time window of the measure.
        """
        return self.beta_multiplier[typ] if self._in_window(t) else 1.0


class UpperBoundCasesBetaMultiplier(BetaMultiplierMeasure):

    def __init__(self, t_window, beta_multiplier, max_pos_tests_per_week=50, intervention_times=None):
        """
        Additional parameters:
        ----------------------
        max_pos_test_per_week : int
            If the number of positive tests per week exceeds this number the measure becomes active
        intervention_times : list of floats
            List of points in time at which interventions can be changed. If 'None' interventions can be changed at any time
        """

        super().__init__(t_window, beta_multiplier)
        self.max_pos_tests_per_week = max_pos_tests_per_week
        self.intervention_history = []
        if intervention_times is not None:
            self.intervention_times = np.asarray(intervention_times)
        else:
            self.intervention_times = None

    def _are_cases_above_threshold(self, t, t_pos_tests):
        # If measures can be changed continuously
        if self.intervention_times is None:
            t_intervention = t
        else:  # If measures can be changed at intervention times
            # Find largest time in intervention_times s.t. t > time
            t_intervention = np.where(t - self.intervention_times > 0, t - self.intervention_times, np.inf).min()

        # Count positive tests in last 7 days from last intervention time
        tmin = t_intervention - 7 * 24
        num_pos_tests = np.sum(np.greater(t_pos_tests, tmin) * np.less(t_pos_tests, t_intervention))
        is_measure_active = num_pos_tests > self.max_pos_tests_per_week
        self.intervention_history.append((t, is_measure_active))
        return is_measure_active

    def beta_factor(self, *, typ, t, t_pos_tests):
        """Returns the multiplicative factor for site type `typ` at time `t`. The
        factor is one if `t` is not in the active time window of the measure.
        """
        if not self._in_window(t):
            return 1.0

        is_measure_active = self._are_cases_above_threshold(t, t_pos_tests)
        return self.beta_multiplier[typ] if is_measure_active else 1.0


"""
========================== INDIVIDUAL COMPLIANCE WITH TRACKING ===========================
"""

class ComplianceForAllMeasure(Measure):
    """
    Compliance measure. All the population has a probability of not using tracking app. This
    influences the ability of smart tracing to track contacts. Each individual uses a tracking
    app with some probability.
    """

    def __init__(self, t_window, p_compliance):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        p_compliance : float
            Probability that individual is compliant, should be in [0,1]
        """
        # Init time window
        super().__init__(t_window)

        # Init probability of respecting measure
        if (not isinstance(p_compliance, float)) or (p_compliance < 0):
            raise ValueError("`compliance` should be a non-negative float")
        self.p_compliance = p_compliance

    def init_run(self, n_people):
        """Init the measure for this run by sampling the compliance of each individual

        Parameters
        ----------
        n_people : int
            Number of people in the population
        """
        # Sample the outcome of the measure for each individual
        self.bernoulli_compliant = np.random.binomial(1, self.p_compliance, size=(n_people))
        self._is_init = True

    @enforce_init_run
    def is_compliant(self, *, j, t):
        """Indicate if individual `j` is compliant 
        """
        return self.bernoulli_compliant[j] and self._in_window(t)
    
    def is_compliant_prob(self, *, j, t):
        """Returns probability of compliance for individual `j` at time `t`
        """
        if self._in_window(t):
            return self.p_compliance
        return 0.0
    
    

"""
=========================== OTHERS ===========================
"""


class TestMeasure(Measure):

    def __init__(self, t_window, tests_per_hour):
        super().__init__(t_window)

    def iter_batch(self):
        """Iterator over the next batch of `tests_per_hour` individuals to test
        according to priority list policy
        """
        #TODO: wait for Manuel's smart test feature


class MeasureList:

    def __init__(self, measure_list):
        self.measure_dict = defaultdict(InterLap)
        for measure in measure_list:
            mtype = type(measure)
            if not issubclass(mtype, Measure):
                raise ValueError(("Measures must instance of subclasses of"
                                  " `Measure` objects"))
            # Add the measure in InterLap format: (t_start, t_end, extra_args)
            self.measure_dict[mtype].update([
                (measure.t_window.left, measure.t_window.right - EPS, measure)
            ])

    def init_run(self, measure_type, **kwargs):
        """Call init_run to all measures of type `measure_type` with the given
        arguments in `kwargs`"""
        for _, _, m in self.measure_dict[measure_type]:
            m.init_run(**kwargs)

    def find(self, measure_type, t):
        """Find, if any, the active measure of `type measure_type` at time `t`
        """
        active_measures = list(self.measure_dict[measure_type].find((t, t)))
        assert len(active_measures) <= 1, ("There cannot be more than one"
                                           "active measure of a given type at"
                                           "once")
        if len(active_measures) > 0:
             # Extract active measure from interlap tuple
            return active_measures[0][2]
        return None  # No active measure

    def is_contained(self, measure_type, t, **kwargs):
        m = self.find(measure_type, t)
        if m is not None:  # If there is an active measure
            # FIXME: time is checked twice, both filtered in the list, and in the is_valid query, not a big problem though...
            return m.is_contained(t=t, **kwargs)
        return False  # No active measure

    def is_compliant(self, measure_type, t, **kwargs):
        m = self.find(measure_type, t)
        # If there is an active compliance measure, 
        # not necessarily related to containment
        if m is not None:  
            return m.is_compliant(t=t, **kwargs)
        return False  # No active compliance measure
    
    def start_containment(self, measure_type, t, **kwargs):
        m = self.find(measure_type, t)
        if m is not None:
            return m.start_containment(t=t, **kwargs)
        return False
    
    def is_contained_prob(self, measure_type, t, **kwargs):
        m = self.find(measure_type, t)
        if m is not None:
            return m.is_contained_prob(t=t, **kwargs)
        return False
    
if __name__ == "__main__":

    # Test SocialDistancingForAllMeasure with p_stay_home=1
    m = SocialDistancingForAllMeasure(t_window=Interval(1.0, 2.0), p_stay_home=1.0)
    m.init_run(n_people=2, n_visits=10)
    assert m.is_contained(j=0, j_visit_id=0, t=0.9) == False
    assert m.is_contained(j=0, j_visit_id=0, t=1.0) == True
    assert m.is_contained(j=0, j_visit_id=0, t=1.1) == True
    assert m.is_contained(j=0, j_visit_id=0, t=2.0) == False

    # Test SocialDistancingForAllMeasure with p_stay_home=0
    m = SocialDistancingForAllMeasure(t_window=Interval(1.0, 2.0), p_stay_home=0.0)
    m.init_run(n_people=2, n_visits=10)
    assert m.is_contained(j=0, j_visit_id=0, t=0.9) == False
    assert m.is_contained(j=0, j_visit_id=0, t=1.0) == False
    assert m.is_contained(j=0, j_visit_id=0, t=1.1) == False
    assert m.is_contained(j=0, j_visit_id=0, t=2.0) == False

    # Test SocialDistancingForAllMeasure with p_stay_home=0.5
    m = SocialDistancingForAllMeasure(t_window=Interval(1.0, 2.0), p_stay_home=0.5)
    m.init_run(n_people=2, n_visits=10000)
    # in window
    mean_at_home = np.mean([m.is_contained(j=0, j_visit_id=i, t=1.1)
                              for i in range(10000)])
    assert abs(mean_at_home - 0.5) < 0.01
    # same but not in window
    mean_at_home = np.mean([m.is_contained(j=0, j_visit_id=i, t=0.9)
                              for i in range(10000)])
    assert mean_at_home == 0.0

   # Test SocialDistancingForPositiveMeasure
    m = SocialDistancingForPositiveMeasure(t_window=Interval(1.0, 2.0), p_stay_home=1.0)
    m.init_run(n_people=2, n_visits=10)
    state_posi_started_at = np.array([0.95, np.inf])
    state_posi_ended_at = np.inf * np.ones(2)
    state_resi_started_at = np.inf * np.ones(2)
    state_dead_started_at = np.inf * np.ones(2)

    assert m.is_contained(j=0, j_visit_id=0, t=0.9, state_posi_started_at=state_posi_started_at,
                          state_posi_ended_at=state_posi_ended_at, state_resi_started_at=state_resi_started_at, 
                          state_dead_started_at=state_dead_started_at) == False
    assert m.is_contained(j=0, j_visit_id=0, t=1.0, state_posi_started_at=state_posi_started_at,
                          state_posi_ended_at=state_posi_ended_at, state_resi_started_at=state_resi_started_at,
                          state_dead_started_at=state_dead_started_at) == True

    state_posi_started_at = np.inf * np.ones(2)
    assert m.is_contained(j=0, j_visit_id=0, t=0.9, state_posi_started_at=state_posi_started_at,
                          state_posi_ended_at=state_posi_ended_at, state_resi_started_at=state_resi_started_at,
                          state_dead_started_at=state_dead_started_at) == False
    assert m.is_contained(j=0, j_visit_id=0, t=1.0, state_posi_started_at=state_posi_started_at,
                          state_posi_ended_at=state_posi_ended_at, state_resi_started_at=state_resi_started_at, 
                          state_dead_started_at=state_dead_started_at) == False

    # Text BetaMultiplierMeasure
    m = BetaMultiplierMeasureBySite(t_window=Interval(1.0, 2.0), beta_multiplier={0: 2.0, 1: 0.0})
    assert m.beta_factor(k=0, t=0.9) == 1.0
    assert m.beta_factor(k=0, t=1.0) == 2.0
    assert m.beta_factor(k=1, t=0.9) == 1.0
    assert m.beta_factor(k=1, t=1.0) == 0.0

    # Test MeasureList
    list_of_measures = [
        BetaMultiplierMeasureBySite(t_window=Interval(1.0, 2.0), beta_multiplier={0: 2.0, 1: 0.0}),
        BetaMultiplierMeasureBySite(t_window=Interval(2.0, 5.0), beta_multiplier={0: 2.0, 1: 0.0}),
        BetaMultiplierMeasureBySite(t_window=Interval(8.0, 10.0), beta_multiplier={0: 2.0, 1: 0.0}),
        SocialDistancingForPositiveMeasure(t_window=Interval(1.0, 2.0), p_stay_home=1.0),
        SocialDistancingForPositiveMeasure(t_window=Interval(2.0, 5.0), p_stay_home=1.0),
        SocialDistancingForPositiveMeasure(t_window=Interval(6.0, 10.0), p_stay_home=1.0),
    ]
    obj = MeasureList(list_of_measures)
    obj.init_run(SocialDistancingForPositiveMeasure, n_people=2, n_visits=10)
    assert obj.find(BetaMultiplierMeasureBySite, t=1.0) == list_of_measures[0]
    assert obj.find(BetaMultiplierMeasureBySite, t=2.0) == list_of_measures[1]
    assert obj.find(BetaMultiplierMeasureBySite, t=5.0) == None
    assert obj.find(SocialDistancingForPositiveMeasure, t=5.0) == None
    assert obj.find(SocialDistancingForPositiveMeasure, t=6.0) == list_of_measures[-1]
