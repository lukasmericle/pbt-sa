import time
import numpy as np
from scipy.special import expit, logit


class SAWorker(object):


    def __init__(self, seed, index, instance, inits, history_horizon,
                     item_values, item_weights, knapsacks):

        # As the subroutine is spawned as a child of another process, it inherits
        # the state of the random number generator of its parent. Without this
        # line, all initializations will be equal across all worker subprocesses.
        np.random.seed(seed)

        # name output files
        self._index = index
        self._instance = instance

        # the constraints of the problem
        self.item_values = item_values
        self.item_weights = item_weights
        self.knapsacks = knapsacks

        # the hyperparameters and parameters
        self.temperature = self._init_attr(inits["temperature"])
        self.cooling_rate = self._init_attr(inits["cooling rate"])
        self.p_mutations = self._init_attr(inits["p mutations"])
        self.n_steps = 0
        self.solution = np.random.binomial(n=1, p=np.random.rand(), size=len(item_values))

        # the other interesting values to keep track of
        self.value = self._get_value(self.solution)
        self.last_values = np.array([self.value for _ in range(history_horizon)])


    def summary_vector(self):
        return [
            self.n_steps, self.value, self.temperature, self.cooling_rate,
            self.p_mutations
        ]


    def csv_row(self):
        """Generate CSV row for recording history."""
        return ",".join(map(str, self.summary_vector())) + "\n"


    def reset(self, template):
        """Copy parameters and hyperparameters from `template`."""
        self.n_steps = 0
        self.temperature = template.temperature
        self.cooling_rate = template.cooling_rate
        self.p_mutations = template.p_mutations
        self.solution = template.solution.copy()
        self._update_value_history()


    def step(self):
        """One SA step."""
        candidate = self._neighbour()
        #if self._accept_candidate_stochastic(candidate):
        if self._accept_candidate_greedy(candidate):
            self.solution = candidate.copy()
        self.n_steps += 1
        self.temperature *= 1 - self.cooling_rate
        self._update_value_history()


    # Helper functions for `explore` step of PBT


    def perturb_temp(self, scale):
        self.temperature *= 1 + scale * np.random.randn()


    def perturb_cooling_rate(self, scale):
        self.cooling_rate *= 1 + scale * np.random.randn()


    def perturb_p_mutations(self, scale):
        self.p_mutations *= 1 + scale * np.random.randn()
        self.p_mutations = np.clip(self.p_mutations, 0, 1)


    # ------


    def _init_attr(self, tup):
        """Generate random numbers in this subprocess
        since passing `np.random.rand()` through the pickling process
        results in a single float."""
        dist = tup[0]
        if dist=='uniform':
            lo = tup[1]
            hi = tup[2]
            return np.random.uniform(lo, hi)
        elif dist=='unilog':
            lo = tup[1]
            hi = tup[2]
            return np.power(10, np.random.uniform(lo, hi))
        elif dist=='normal':
            mu = tup[1]
            sigma = tup[2]
            return np.random.normal(mu, sigma)
        elif dist=='expit':
            lo = logit(tup[1])
            hi = logit(tup[2])
            return expit(np.random.uniform(lo, hi))
        elif dist=='const':
            c = tup[1]
            return c


    def _neighbour(self):
        excluded = np.where(self.solution==0)[0]
        new_inclusions = np.random.binomial(n=1, p=self.p_mutations, size=len(excluded)).astype(bool)
        added_items = excluded[new_inclusions]
        candidate = self.solution.copy()
        candidate[added_items] = 1
        candidate = self._obey_constraints(candidate)
        return candidate


    def _obey_constraints(self, candidate):
        """Removes random items until constraints are satisfied."""
        included = np.where(candidate==1)[0]
        included = np.random.permutation(included)
        allocations = self._get_allocations(candidate)
        c = 0
        while np.any(allocations > self.knapsacks):
            item = included[c]
            allocations -= self.item_weights[item]
            candidate[item] = 0
            c += 1
        return candidate


    def _accept_candidate_stochastic(self, candidate):
        """Accept candidate conditional on Metropolis-Hastings acceptance probability."""
        arg = (self._get_value(candidate) - self._get_value(self.solution))/self.temperature
        # if arg is positive, will always accept
        # if arg is negative, accept with certain probability
        if abs(arg) > 700:
            return arg >= 0  # otherwise, we get a `RuntimeWarning` for overflow in the next line
        p = np.exp(arg)
        return np.random.rand() < p


    def _accept_candidate_greedy(self, candidate):
        """Accept candidate conditional on Metropolis-Hastings acceptance probability."""
        if self._get_value(candidate) >= self._get_value(self.solution):
            return True
        return False


    def _update_value_history(self):
        self.last_values[:-1] = self.last_values[1:]
        self.last_values[-1] = self._get_value(self.solution)
        self.value = self.last_values[-1]


    def _get_value(self, candidate):
        """Return the value of the solution."""
        total_value = self._get_values(candidate).sum()
        allocations = self._get_allocations(candidate)
        if np.any(allocations > self.knapsacks):
            return 0
        return total_value


    def _get_allocations(self, candidate):
        """Apply solution to get knapsack allocations."""
        return self._get_weights(candidate).sum(axis=0)


    def _get_values(self, candidate):
        """Get values of allocated items."""
        return self.item_values[candidate.astype(bool)]


    def _get_weights(self, candidate):
        """Get weights of allocated items."""
        return self.item_weights[candidate.astype(bool)]
