import math, random

"""
    Represents dynamic programming table entries, storing the argument
    and the corresponding table value. The purpose of the class is to make
    computing the minimum, while retaining both the min and argmin a bit
    cleaner.
"""
class DynamicValue(object):
    def __init__(self, arg, val):
        self.arg = arg
        self.val = val
    def __lt__(self, other):
        return self.val < other.val
    def __eq__(self, other):
        return self.val == other.val
    def __add__(self, other):
        # ORDERING OF OPERANDS IS VERY IMPORTANT
        return DynamicValue(self.arg + other.arg, \
                            self.val + other.val)
    def __repr__(self):
        return "(" + repr(self.arg) + ", " + repr(self.val) + ")"
    def __str__(self):
        return self.__repr__()

"""
    Implementation of memoization.
"""
def memoize(f):
    vals = {}
    def memoized(*args):
        if args not in vals:
            vals[args] = f(*args)
        return vals[args]
    return memoized

"""
    Performs the Viterbi algorithm. Assumes one observation per state.

    states - Labels for the different HMM states (list)
    observations - Observations from each state (list)
    P - The probability transition matrix of the states (nested dicts)
    Q - The observation probability matrix (nested dicts)
"""
def viterbi(states, observations, P, Q, prior):
    # Calculate's the distance between states -ln(P(x,x')*Q(x',y')).
    # If this is the initial "distance" at the start state, calculates
    # -ln(prior(x_0)*Q(x_0,y_0)).
    def dist(m, x, x_prime=None):
        if x_prime:
            val = - math.log(Q[x_prime][observations[m]] * P[x][x_prime])
            return DynamicValue([x_prime], val)
        else:
            val = - math.log(Q[x][observations[m]] * prior[x])
            return DynamicValue([x], val)

    # The dynamic program algorithm for Viterbi
    @memoize
    def V(m, x):
        # If we've reached the penultimate state, don't recurse
        if m < len(observations) - 2:
            return min([dist(m+1,x,x_prime) + V(m+1,x_prime) for x_prime in states])
        else:
            return min([dist(m+1,x,x_prime) for x_prime in states])
    return min([dist(0,x) + V(0,x) for x in states]).arg

"""
    Samples a distribution.
"""
def sample(distribution):
    values = distribution.keys()
    choice = random.random()
    i, total= 0, distribution[values[0]]
    while choice > total:
        i += 1
        total += distribution[values[i]]
    return values[i]

"""
    Generates a state sequence and observations based off of the given
    probability distributions.
"""
def generate_seq(states, obs_set, P, Q, prior, n):
    state_seq = [sample(prior)]
    observations = [sample(Q[state_seq[-1]])]
    for i in range(n-1):
        state_seq += [sample(P[state_seq[-1]])]
        observations += [sample(Q[state_seq[-1]])]
    return (state_seq, observations)

"""
    Calculates the error between the actual state sequence and the
    estimated sequence, via the percent of states wrong.
"""
def calc_error(state_seq, est_seq):
    assert len(state_seq) == len(est_seq), \
           "Sequences must be the same length!"
    wrong = 0.0
    for act, est in zip(state_seq, est_seq):
        if act != est: wrong += 1
    return wrong/len(state_seq)


def example():
    # In this example, we assume two states, with observations modeled
    # as a binary symmetric channel with error epsilon
    states = ['a', 'b']
    obs_set = {0,1}
    alpha = 0.5
    epsilon = 0.5
    P = {'a': {'a': 1 - alpha, 'b': alpha },\
         'b': {'a': alpha, 'b': 1 - alpha }}
    Q = {'a': { 0 : 1 - epsilon, 1 : epsilon },\
         'b': { 0 : epsilon, 1 : 1 - epsilon }}
    prior = { 'a' : 0.5, 'b' : 0.5 }
    n = 10

    state_seq, observations = generate_seq(states,obs_set,P,Q,prior,n)
    print state_seq, observations
    est_seq = viterbi(states, observations, P, Q, prior)

    print "Actual sequence: " + repr(state_seq)
    print "Estimated sequence: " + repr(est_seq)
    print "Error: " + repr(calc_error(state_seq, est_seq))

if __name__ == '__main__':
    example()
    exit()
