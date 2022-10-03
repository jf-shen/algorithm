import numpy as np

"""
Given lb = [l1, l2, ..., ln], ub = [u1, u2, ..., un], the sampler uniformly samples a distribution p = [p1, p2, ..., pn] that satisfies:
1. lb <= p <= ub 
2. sum(p) = 1 
The sampling is implemented by Gibbs sampling.
"""


class GibbsSampler:
    def __init__(self, init_state, param_lb, param_ub):

        self.param_lb = np.array(param_lb)
        self.param_ub = np.array(param_ub)
        self.param_num = len(param_lb)

        # check validity
        assert len(param_lb) == len(param_ub), "len(param_lb) = " + str(param_lb) + ", len(param_ub) = " + str(param_ub)
        assert np.all((self.param_lb >= 0) & (self.param_lb <= 1)), "param_lb has values out of [0,1]"
        assert np.all((self.param_ub >= 0) & (self.param_ub <= 1)), "param_ub has values out of [0,1]"
        assert np.all(self.param_lb <= self.param_ub), "param_lb > param_ub"
        assert np.sum(param_lb) <= 1, "sum(param_lb) > 1"
        assert np.sum(param_ub) >= 1, "sum(param_ub) < 1"

        self.step = 0
        self.state = init_state

        assert np.all((self.state >= self.param_lb) & (self.state <= self.param_ub)), "init_state out of bound"
        assert np.sum(self.state) == 1, "sum(init_sate) != 1"

    def transit(self, steps=None, verbose=False):
        if steps is None:
            steps = 1

        if verbose:
            print('step 0:', np.round(self.state, 6))

        for i in range(steps):
            idx = np.random.choice(self.param_num - 1)

            x = self.state[idx]
            y = self.state[-1]

            s = x + y
            x_lb = max(self.param_lb[idx], s - self.param_ub[-1])
            x_ub = min(self.param_ub[idx], s - self.param_lb[-1])

            new_x = x_lb + np.random.uniform() * (x_ub - x_lb)
            new_y = s - new_x

            self.state[idx] = new_x
            self.state[-1] = new_y
            self.step += 1

            if verbose:
                print('step %i:' % (i + 1), np.round(self.state, 6), ', idx = ', idx)

    def sample(self):
        return self.state.copy()