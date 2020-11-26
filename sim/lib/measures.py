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

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self.bernoulli_stay_home = None
            self._is_init = False


class UpperBoundCasesSocialDistancing(SocialDistancingForAllMeasure):

    def __init__(self, t_window, p_stay_home, max_pos_tests_per_week_per_100k, intervention_times=None, init_active=False):
        """
        Additional parameters:
        ----------------------
        max_pos_test_per_week : int
            If the number of positive tests per week exceeds this number the measure becomes active
        intervention_times : list of floats
            List of points in time at which measures can become active. If 'None' measures can be changed at any time
        """

        super().__init__(t_window, p_stay_home)
        self.max_pos_tests_per_week_per_100k = max_pos_tests_per_week_per_100k
        self.intervention_times = intervention_times
        self.intervention_history = InterLap()
        if init_active:
            self.intervention_history.update([(t_window.left, t_window.left + 7 * 24 - EPS, True)])

    def init_run(self, n_people, n_visits):
        super().init_run(n_people, n_visits)
        self.scaled_test_threshold = self.max_pos_tests_per_week_per_100k / 100000 * n_people

    def _is_measure_active(self, t, t_pos_tests):
        # If measures can only become active at 'intervention_times'
        if self.intervention_times is not None:
            # Find largest 'time' in intervention_times s.t. t > time
            intervention_times = np.asarray(self.intervention_times)
            idx = np.where(t - intervention_times > 0, t - intervention_times, np.inf).argmin()
            t = intervention_times[idx]

        t_in_history = list(self.intervention_history.find((t, t)))
        if t_in_history:
            is_active = t_in_history[0][2]
        else:
            is_active = self._are_cases_above_threshold(t, t_pos_tests)
            if is_active:
                self.intervention_history.update([(t, t + 7 * 24 - EPS, True)])
        return is_active

    def _are_cases_above_threshold(self, t, t_pos_tests):
        # Count positive tests in last 7 days from last intervention time
        tmin = t - 7 * 24
        num_pos_tests = np.sum(np.greater(t_pos_tests, tmin) * np.less(t_pos_tests, t))
        is_above_threshold = num_pos_tests > self.scaled_test_threshold
        return is_above_threshold

    @enforce_init_run
    def is_contained(self, *, j, j_visit_id, t, t_pos_tests):
        """Indicate if individual `j` respects measure for visit `j_visit_id`
        """
        if not self._in_window(t):
            return False

        is_home_now = self.bernoulli_stay_home[j, j_visit_id]
        return is_home_now and self._is_measure_active(t, t_pos_tests)

    @enforce_init_run
    def is_contained_prob(self, *, j, t, t_pos_tests):
        """Returns probability of containment for individual `j` at time `t`
        """
        if not self._in_window(t):
            return 0.0

        if self._is_measure_active(t, t_pos_tests):
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
        return is_home_now and state_dict[self.state_label][j] and self._in_window(t)
    
    @enforce_init_run
    def is_contained_prob(self, *, j, t, state_started_at_dict, state_ended_at_dict):
        """Returns probability of containment for individual `j` at time `t`
        """
        if self._in_window(t) and state_started_at_dict[self.state_label][j] <= t <= \
                state_ended_at_dict[self.state_label][j]:
            return self.p_stay_home
        return 0.0


class SocialDistancingBySiteTypeForAllMeasure(Measure):
    """
    Social distancing measure. All the population is advised to stay home. Each
    visit of each individual respects the measure with some probability.
    """

    def __init__(self, t_window, p_stay_home_dict):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        p_stay_home_dict : dict of site_type : float
            Probability of respecting the measure for a given site type, should be in [0,1]
        """
        # Init time window
        super().__init__(t_window)

        # Init probabilities of respecting measure
        if (not isinstance(p_stay_home_dict, dict)) or any([p < 0.0 or p > 1.0 for p in p_stay_home_dict.values()]):
            raise ValueError("`p_stay_home_dict` should contain non-negative floats between 0 and 1")
        self.p_stay_home_dict = p_stay_home_dict

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
        self.bernoulli_stay_home_type = {
            k : np.random.binomial(1, p, size=(n_people, n_visits))
            for k, p in self.p_stay_home_dict.items()
        }
        self._is_init = True

    @enforce_init_run
    def is_contained(self, *, j, j_visit_id, site_type, t):
        """Indicate if individual `j` respects measure for visit `j_visit_id`
        """
        is_home_now = self.bernoulli_stay_home_type[site_type][j, j_visit_id]
        return is_home_now and self._in_window(t)

    @enforce_init_run
    def is_contained_prob(self, *, j, site_type, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        if self._in_window(t):
            return self.p_stay_home_dict[site_type]
        return 0.0

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self.bernoulli_stay_home = None
            self._is_init = False


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
            return self.p_isolate
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

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self.bernoulli_stay_home = None
            self._is_init = False


class SocialDistancingForSmartTracing(Measure):
    """
    Social distancing measure. Only the population who intersected with positive cases 
    for ``smart_tracing_isolation_duration``. Each visit of each individual respects the measure with 
    some probability.
    """

    def __init__(self, t_window, p_stay_home, smart_tracing_isolation_duration):
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
        self.smart_tracing_isolation_duration = smart_tracing_isolation_duration

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
        self.bernoulli_stay_home = np.random.binomial(1, self.p_stay_home, size=(n_people, n_visits))
        self.intervals_stay_home = [InterLap() for _ in range(n_people)]
        #self.got_contained = np.zeros([n_people, 2])
        # self.got_contained = [[] for _ in range(n_people)]
        self._is_init = True

    @enforce_init_run
    def is_contained(self, *, j, j_visit_id, state_nega_started_at, state_nega_ended_at, t):
        """Indicate if individual `j` respects measure for visit `j_visit_id`
        Negatively tested are not isolated
        """
        is_not_nega_now = not (state_nega_started_at[j] <= t and t < state_nega_ended_at[j])

        if self._in_window(t) and self.bernoulli_stay_home[j, j_visit_id] and is_not_nega_now:
            for interval in self.intervals_stay_home[j].find((t, t)):
                return True
        return False

    @enforce_init_run
    def start_containment(self, *, j, t):
        self.intervals_stay_home[j].update([(t, t + self.smart_tracing_isolation_duration)])
        # self.got_contained[j].append([t, t + self.smart_tracing_isolation_duration])
        return

    @enforce_init_run
    def is_contained_prob(self, *, j, state_nega_started_at, state_nega_ended_at, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        is_not_nega_now = not (state_nega_started_at[j] <= t and t < state_nega_ended_at[j])

        if self._in_window(t) and is_not_nega_now:
            for interval in self.intervals_stay_home[j].find((t, t)):
                return self.p_stay_home
        return 0.0

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self.bernoulli_stay_home = None
            self._is_init = False


class SocialDistancingSymptomaticAfterSmartTracing(Measure):
    """
    Social distancing measure. If an individual develops symptoms after they were identified
    and isolated by contact tracing, the individual isolates until symptoms disappear.
    """

    def __init__(self, t_window, p_stay_home, smart_tracing_isolation_duration):
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
        self.smart_tracing_isolation_duration = smart_tracing_isolation_duration

    def init_run(self, n_people):
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
        self.bernoulli_stay_home = np.random.binomial(1, self.p_stay_home, size=n_people)
        self.got_contained = np.zeros(n_people, dtype='bool')
        self._is_init = True

    @enforce_init_run
    def is_contained(self, *, j, state_isym_started_at, state_isym_ended_at, state_nega_started_at, state_nega_ended_at, t):
        """Indicate if individual `j` respects measure
        Negatively tested are not isolated.
        """
        is_not_nega_now = not (state_nega_started_at[j] <= t and t < state_nega_ended_at[j])
        is_isym_now = (state_isym_started_at[j] <= t and t < state_isym_ended_at[j])

        return self._in_window(t) and self.bernoulli_stay_home[j] and is_isym_now and is_not_nega_now and self.got_contained[j]

    @enforce_init_run
    def start_containment(self, *, j, t):
        self.got_contained[j] = True
        return

    @enforce_init_run
    def is_contained_prob(self, *, j, state_isym_started_at, state_isym_ended_at, state_nega_started_at, state_nega_ended_at, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        is_not_nega_now = not (state_nega_started_at[j] <= t and t < state_nega_ended_at[j])
        is_isym_now = (state_isym_started_at[j] <= t and t < state_isym_ended_at[j])

        if self._in_window(t) and is_isym_now and self.got_contained[j] and is_not_nega_now:
            return self.p_stay_home
        return 0.0

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self.bernoulli_stay_home = None
            self.got_contained = None
            self._is_init = False

class SocialDistancingForSmartTracingHousehold(Measure):
    """
    Social distancing measure. Isolate traced individuals cases from household members. 
    Only the population who intersected with positive cases for ``smart_tracing_isolation_duration``. 
    Each visit of each individual respects the measure with some probability.
    """

    def __init__(self, t_window, p_isolate, smart_tracing_isolation_duration):
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

        # Init probability of respecting measure
        if (not isinstance(p_isolate, float)) or (p_isolate < 0):
            raise ValueError("`p_isolate` should be a non-negative float")
        self.p_isolate = p_isolate
        self.smart_tracing_isolation_duration = smart_tracing_isolation_duration

    def init_run(self, n_people):
        """Init the measure for this run. Sampling of Bernoulli of respecting the measure done online."""
        self.intervals_isolated = [InterLap() for _ in range(n_people)]
        self._is_init = True

    @enforce_init_run
    def is_contained(self, *, j, state_nega_started_at, state_nega_ended_at, t):
        """Indicate if individual `j` respects measure at time `t`
        """
        is_not_nega_now = not (state_nega_started_at[j] <= t and t < state_nega_ended_at[j])
        is_isolated = np.random.binomial(1, self.p_isolate)
        if self._in_window(t) and is_isolated and is_not_nega_now:
            for interval in self.intervals_isolated[j].find((t, t)):
                return True
        return False

    @enforce_init_run
    def start_containment(self, *, j, t):
        self.intervals_isolated[j].update([(t, t + self.smart_tracing_isolation_duration)])
        return

    @enforce_init_run
    def is_contained_prob(self, *, j, state_nega_started_at, state_nega_ended_at, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        is_not_nega_now = not (state_nega_started_at[j] <= t and t < state_nega_ended_at[j])

        if self._in_window(t) and is_not_nega_now:
            for interval in self.intervals_isolated[j].find((t, t)):
                return self.p_isolate
        return 0.0

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self._is_init = False


class SocialDistancingSymptomaticAfterSmartTracingHousehold(Measure):

    """
    Social distancing measure. If an individual develops symptoms after they were identified
    and isolated by contact tracing, the individual isolates from household members
    until symptoms disappear.
    """

    def __init__(self, t_window, p_isolate, smart_tracing_isolation_duration):
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

        # Init probability of respecting measure
        if (not isinstance(p_isolate, float)) or (p_isolate < 0):
            raise ValueError("`p_isolate` should be a non-negative float")
        self.p_isolate = p_isolate
        self.smart_tracing_isolation_duration = smart_tracing_isolation_duration

    def init_run(self, n_people):
        """Init the measure for this run. Sampling of Bernoulli of respecting the measure done online."""
        # Sample the outcome of the measure for each visit of each individual
        self.got_contained = np.zeros(n_people, dtype='bool')
        self._is_init = True

    @enforce_init_run
    def is_contained(self, *, j, state_isym_started_at, state_isym_ended_at, state_nega_started_at, state_nega_ended_at, t):
        """Indicate if individual `j` respects measure
        """
        is_isolated = np.random.binomial(1, self.p_isolate)
        is_isym_now = (state_isym_started_at[j] <= t and t < state_isym_ended_at[j])
        is_not_nega_now = not (state_nega_started_at[j] <= t and t < state_nega_ended_at[j])

        return self._in_window(t) and is_isolated and is_isym_now and self.got_contained[j] and is_not_nega_now

    @enforce_init_run
    def start_containment(self, *, j, t):
        self.got_contained[j] = True
        return

    @enforce_init_run
    def is_contained_prob(self, *, j, state_isym_started_at, state_isym_ended_at, state_nega_started_at, state_nega_ended_at, t):
        """Returns probability of containment for individual `j` at time `t`
        """
        is_isym_now = (state_isym_started_at[j] <= t and t < state_isym_ended_at[j])
        is_not_nega_now = not (state_nega_started_at[j] <= t and t < state_nega_ended_at[j])

        if self._in_window(t) and is_isym_now and self.got_contained[j] and is_not_nega_now:
            return self.p_isolate
        return 0.0

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self.got_contained = None
            self._is_init = False

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


class APrioriBetaMultiplierMeasureByType(Measure):

    def __init__(self, beta_multiplier):
        """

        Parameters
        ----------
        beta_multiplier : list of floats
            List of multiplicative factor to infection rate at each site
        """

        super().__init__(Interval(0.0, np.inf))
        if (not isinstance(beta_multiplier, dict)
                or (min(beta_multiplier.values()) < 0)):
            raise ValueError(("`beta_multiplier` should be dict of"
                              " non-negative floats"))
        self.beta_multiplier = beta_multiplier

    def beta_factor(self, *, typ):
        """Returns the multiplicative factor for site type `typ` independent of time `t`
        """
        return self.beta_multiplier[typ]


class UpperBoundCasesBetaMultiplier(BetaMultiplierMeasure):

    def __init__(self, t_window, beta_multiplier, max_pos_tests_per_week_per_100k, intervention_times=None, init_active=False):
        """
        Additional parameters:
        ----------------------
        max_pos_test_per_week : int
            If the number of positive tests per week exceeds this number the measure becomes active
        intervention_times : list of floats
            List of points in time at which interventions can be changed. If 'None' interventions can be changed at any time
        init_active : bool
            If true measure is active in the first week of the simulation when there are no test counts yet
        """

        super().__init__(t_window, beta_multiplier)
        self.max_pos_tests_per_week_per_100k = max_pos_tests_per_week_per_100k
        self.intervention_times = intervention_times
        self.intervention_history = InterLap()
        if init_active:
            self.intervention_history.update([(t_window.left, t_window.left + 7 * 24 - EPS, True)])

    def init_run(self, n_people, n_visits):
        self.scaled_test_threshold = self.max_pos_tests_per_week_per_100k / 100000 * n_people
        self._is_init = True

    @enforce_init_run
    def _is_measure_active(self, t, t_pos_tests):
        # If measures can only become active at 'intervention_times'
        if self.intervention_times is not None:
            # Find largest 'time' in intervention_times s.t. t > time
            intervention_times = np.asarray(self.intervention_times)
            idx = np.where(t - intervention_times > 0, t - intervention_times, np.inf).argmin()
            t = intervention_times[idx]

        t_in_history = list(self.intervention_history.find((t, t)))
        if t_in_history:
            is_active = t_in_history[0][2]
        else:
            is_active = self._are_cases_above_threshold(t, t_pos_tests)
            if is_active:
                self.intervention_history.update([(t, t+7*24 - EPS, True)])
        return is_active

    @enforce_init_run
    def _are_cases_above_threshold(self, t, t_pos_tests):
        # Count positive tests in last 7 days from last intervention time
        tmin = t - 7 * 24
        num_pos_tests = np.sum(np.greater(t_pos_tests, tmin) * np.less(t_pos_tests, t))
        is_above_threshold = num_pos_tests > self.scaled_test_threshold
        return is_above_threshold

    @enforce_init_run
    def beta_factor(self, *, typ, t, t_pos_tests):
        """Returns the multiplicative factor for site type `typ` at time `t`. The
        factor is one if `t` is not in the active time window of the measure.
        """
        if not self._in_window(t):
            return 1.0

        is_measure_active = self._is_measure_active(t, t_pos_tests)
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

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self.bernoulli_compliant = None
            self._is_init = False
    

class ManualTracingForAllMeasure(Measure):
    """
    Participation measure. All the population has a probability of participating in
    manual tracing. This influences the ability of smart tracing to track contacts. 
    
    If an individual i complies with this measure, contacts which
    i)   happen at sites that i recalls with probability `p_recall`
    ii)  happen at sites that have a Bluetooth beacon

    can be traced, even if i does not comply with contact tracing itself.
    """

    def __init__(self, t_window, p_participate, p_recall):
        """

        Parameters
        ----------
        t_window : Interval
            Time window during which the measure is active
        p_participate : float
            Probability that individual is participating with manual contact tracing, should be in [0,1]
        p_recall : float
            Probability that individual recalls a given visit, should be in [0,1]
        """
        # Init time window
        super().__init__(t_window)

        # Init probability of respecting measure
        if (not isinstance(p_participate, float)) or (p_participate < 0):
            raise ValueError("`p_participate` should be a non-negative float")
        if (not isinstance(p_recall, float)) or (p_recall < 0):
            raise ValueError("`p_recall` should be a non-negative float")
        self.p_participate = p_participate
        self.p_recall = p_recall

    def init_run(self, n_people, n_visits):
        """Init the measure for this run by sampling the compliance of each individual

        Parameters
        ----------
        n_people : int
            Number of people in the population
        n_visits : int
            Maximum number of visits of an individual
        """
        # Sample the comliance outcome of the measure for each individual
        self.bernoulli_participate = np.random.binomial(1, self.p_participate, size=(n_people)).astype(np.bool)

        # Sample the site recall outcome for each visit of each individual
        self.bernoulli_recall = np.random.binomial(1, self.p_recall, size=(n_people, n_visits)).astype(np.bool)
        self._is_init = True

    @enforce_init_run
    def is_active(self, *, j, t, j_visit_id):
        """
        j : int
            individual
        t : float
            time
        j_visit_id : int (optional)
            visit id

        If j_visit_id == None:
            Returns True iff `j` is compliant with manual tracing
        else:
            Returns True iff `j` is compliant with manual tracing _and_ recalls visit `j_visit_id`
        """
        if j_visit_id is None:
            return self.bernoulli_participate[j] and self._in_window(t)
        else:
            return self.bernoulli_recall[j, j_visit_id] and self.bernoulli_participate[j] and self._in_window(t)

    def exit_run(self):
        """ Deletes bernoulli array. """
        if self._is_init:
            self.bernoulli_participate = None
            self.bernoulli_recall = None
            self._is_init = False
    

"""
=========================== OTHERS ===========================
"""

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

    def exit_run(self):
        """
        Call exit_run for all measures that use bernoulli arrays to delete those in order to save memory.
        """
        for measure_type in self.measure_dict.keys():
            for _, _, m in self.measure_dict[measure_type]:
                try:
                    m.exit_run()
                except AttributeError:
                    pass

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

    def find_first(self, measure_type):
        """Find, if any, the first active measure of `type measure_type` 
        """
        measures = list(self.measure_dict[measure_type].find((0.0, np.inf)))
        if len(measures) > 0:
             # Extract first measure from interlap tuple
            return measures[0][2]
        return None  # No active measure

    '''Containment-type measures (default: FALSE)
        i.e. if no containment measure found, assume _is not_ contained
    '''
    def is_contained(self, measure_type, t, **kwargs):
        m = self.find(measure_type, t)
        if m is not None:  # If there is an active measure
            # FIXME: time is checked twice, both filtered in the list, and in the is_valid query, not a big problem though...
            return m.is_contained(t=t, **kwargs)
        return False  # No active measure

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

    '''Participation-type measures (default: FALSE)
        i.e. if no participation measure found, assume _is not_ active
    '''
    def is_active(self, measure_type, t, **kwargs):
        m = self.find(measure_type, t)
        if m is not None:  # If there is an active measure
            # FIXME: time is checked twice, both filtered in the list, and in the is_valid query, not a big problem though...
            return m.is_active(t=t, **kwargs)
        return False  # No active measure

    '''Compliance type measures (default: TRUE)
        i.e. if no compliance measure found, assume _is_ compliant
    '''
    def is_compliant(self, measure_type, t, **kwargs):
        m = self.find(measure_type, t)
        # If there is an active compliance measure, 
        # not necessarily related to containment
        if m is not None:  
            return m.is_compliant(t=t, **kwargs)
        return True  # No active compliance measure
    

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
