"""
Reference:
Stochastic optimal control theory, H.J. Kappen， P6
"""


def argmax(domain, fn):
    max_x = None
    max_y = None
    for x in domain:
        y = fn(x)
        if (max_x is None) or (y > max_y):
            max_x = x
            max_y = y
    return max_x


class DynamicProgramming:
    def __init__(self, state_space, action_space, trans_fn, reward_fn, final_reward_fn=None):
        """
        Args:
            @T: time length [int]
            @state_space: [list<state>]
            @action_space: [list<action>]
            @trans_fn: (t, x, action) -> state   [(int, state, action) -> state]
            @reward_fn: (t, x, action) -> reward [(int, state, action) -> float]
            @final_reward_fn: final_x -> reward  [state -> float]
        """

        self.state_space = state_space
        self.action_space = action_space

        self.trans_fn = trans_fn
        self.reward_fn = reward_fn
        self.final_reward_fn = final_reward_fn

        self.J = None
        self.action_path = None
        self.state_path = None

        self.is_run = False

    def run(self, T):

        """ set param """
        state_space = self.state_space
        action_space = self.action_space

        trans_fn = self.trans_fn
        reward_fn = self.reward_fn
        final_reward_fn = self.final_reward_fn

        if final_reward_fn is None:
            final_reward_fn = lambda x: reward_fn(T, x, None)

        # reward function is time independent:
        if reward_fn.__code__.co_argcount == 2:
            reward_fn = lambda t, x, action: reward_fn(x, action)

        # transition function is time independent:
        if trans_fn.__code__.co_argcount == 2:
            trans_fn = lambda t, x, action: trans(x, action)

        """ dynamic programming """
        J = dict([(x, final_reward_fn(x)) for x in state_space])  # t = T

        state_path = dict([(x, [x]) for x in state_space])
        action_path = dict([(x, []) for x in state_space])

        for t in range(T - 1, -1, -1):  # t = T-1, ..., 1, 0
            pre_J = dict()
            pre_state_path = dict()
            pre_action_path = dict()
            for x in state_space:
                fn = lambda action: reward_fn(t, x, action) + J[trans_fn(t, x, action)]
                action = argmax(action_space, fn)  # best action at x
                pre_J[x] = fn(action)

                pre_state_path[x] = [x] + state_path[trans_fn(t, x, action)]
                pre_action_path[x] = [action] + action_path[trans_fn(t, x, action)]

            J = pre_J
            state_path = pre_state_path
            action_path = pre_action_path

        self.J = J
        self.state_path = state_path
        self.action_path = action_path

        self.is_run = True

        print("Finish running!")

    def get_optimal_actions(self, state):
        if not self.is_run:
            raise Exception("Dynamic Programming not run, try: dp.run(T=10)")

        return self.action_path[state]

    def get_optimal_states(self, state):
        if not self.is_run:
            raise Exception("Dynamic Programming not run, try: dp.run(T=10)")
        return self.state_path[state]


if __name__ == '__main__':
    state_space = [0, 1]
    action_space = [0, 1]
    trans_fn = lambda t, x, action: action ^ x
    reward_fn = lambda t, x, action: action + x
    final_reward_fn = lambda x: x

    dp = DynamicProgramming(state_space=state_space,
                            action_space=action_space,
                            trans_fn=trans_fn,
                            reward_fn=reward_fn,
                            final_reward_fn=final_reward_fn)

    dp.run(10)
    print("optimal actions started from 0:", dp.get_optimal_actions(0))
    print("optimal state seqence started from 0:", dp.get_optimal_states(0))
