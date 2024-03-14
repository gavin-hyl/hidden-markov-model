import numpy as np

def normalize_distribution(arr: np.array):
    s = sum(arr)
    if not s == 0:
        return arr / s
    else:
        return np.repeat(1/len(arr), len(arr))

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.

    Parameters
        N:          Number of states.
        V:          Number of observations.
        A:          The transition matrix.
        O:          The observation matrix.
        T0:         Starting transition probabilities. The i^th element is the 
                    probability of transitioning from the start state to state i.
                    It is assumed to be uniform.
    '''

    def __init__(self, A=None, O=None, A_start=None, N=1, V=1):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state. There is no integer associated with the 
              start state, only probabilities in the vector A_start.
            - There is no end state.
        
        Arguments:
            N:          The number of hidden states.
            V:          The number of distinct observations.
            A:          Transition matrix with dimensions N x N.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions N x V.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
            A_start:    The starting state's distribution.
        
        If T and O are not both provided, then N and V are used. If A_start is
        not provided, a uniform starting distribution is assumed.
        '''
        if not A is None and not O is None:
            A = np.array(A)
            O = np.array(O)
            self.N = A.shape[0]
            self.V = O.shape[1]
            self.A = A
            self.O = O
        else:
            self.N = N
            self.V = V
            A = np.random.uniform(size=[N, N])
            O = np.random.uniform(size=[N, V])
            self.A = [row / sum(row) for row in A]
            self.O = [row / sum(row) for row in O]
        self.A_start = np.repeat(1/self.N, self.N) if A_start is None else A_start


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            State sequence corresponding to x with the highest probability.
        '''

        L = len(x)

        # seqs[i][j] is the maximum probability state sequence of length i+1 ending in j
        seqs = [[[] for _ in range(self.N)] for _ in range(L)]
        # probs[i][j] is the probability of seqs[i][j]
        probs = np.zeros((L, self.N))

        seqs[0] = [[p] for p in range(self.N)]
        probs[0] = np.array([self.A_start[y] * self.O[y][x[0]] for y in range(self.N)])
        for t in range(1, L):
            for y_next in range(self.N):
                best_prev_seq = seqs[t-1][0]
                best_joint_prob = float('-inf')
                for prev_seq, prev_joint_prob in zip(seqs[t-1], probs[t-1]):
                    y_prev = prev_seq[-1]
                    x_t = x[t]
                    if prev_joint_prob == 0 \
                        or self.A[y_prev][y_next] == 0 \
                        or self.O[y_next][x_t] == 0: # impossible states
                        continue
                    joint_prob = np.log(prev_joint_prob) \
                                + np.log(self.A[y_prev][y_next]) \
                                + np.log(self.O[y_next][x_t])
                    if joint_prob > best_joint_prob:
                        best_joint_prob = joint_prob
                        best_prev_seq = prev_seq
                seqs[t][y_next] = best_prev_seq + [y_next]
                probs[t][y_next] = np.exp(best_joint_prob)

        return seqs[L-1][np.argmax(probs[L-1])]


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.
                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        L = len(x)
        alphas = np.zeros((L+1, self.N))    # alphas[0] will always be [0..0]
        alphas[1] = self.O.T[x[0]] * self.A_start

        for t in range(1, L):
            alpha_t = np.zeros((self.N,))
            for a in range(self.N):
                alpha_t[a] = (alphas[t] @ (self.A.T)[a]) * self.O[a][x[t]]
            if normalize:
                alpha_t /= np.sum(alpha_t)
            alphas[t + 1] = alpha_t
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of beta_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        L = len(x)      # Length of sequence.
        betas = np.ones((L+1, self.N))

        for t in range(L-1, -1, -1):    # L-1, L-2, ..., 0
            beta_b = np.zeros(shape=(self.N,))
            for y_t in range(self.N):
                prob_sum = 0
                for y_next in range(self.N):
                    seq_prob = betas[t+1][y_next]
                    transition_prob = self.A[y_t][y_next]
                    emission_prob = self.O[y_next][x[t]]
                    prob_sum += seq_prob * transition_prob * emission_prob
                beta_b[y_t] = prob_sum
            if normalize:
                beta_b /= np.sum(beta_b)
            betas[t] = beta_b
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        denoms = np.zeros(shape=(self.N,))
        # Clear O and A matrices
        self.O = np.zeros((self.N, self.V))
        self.A = np.zeros((self.N, self.N))
        for y in Y:
            for state, next_state in zip(y[:-1], y[1:]):
                denoms[state] += 1
                self.A[state][next_state] += 1
        for i, (row, denom) in enumerate(zip(self.A, denoms)):
            if denom >= 1:
                row /= denom
            else:
                self.A[i] = np.repeat(1/self.N, self.N)

        # Calculate each element of O using the M-step formulas.
        denoms = np.zeros(shape=(self.N,))
        for x, y in zip(X, Y):
            for obs, state in zip(x, y):
                denoms[state] += 1
                self.O[state][obs] += 1
        for i, (row, denom) in enumerate(zip(self.O, denoms)):
            if denom >= 1:
                row /= denom
            else:
                self.O[i] = np.repeat(1/self.V, self.V)

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of variable-length lists, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            N_iters:    The number of iterations to train on.
        '''

        # E step functions
        def joint_prob_xj(alphas, betas, j):
            '''
            Helper function to calculate the joint probability P(y^j = a, x).
            Returns a vector containing P(y=a_1, x) .. P(y=a_L, x).
            j should range from [1, M]
            '''
            joint_probs = [0 for _ in range(self.N)]
            prob_sum = 0
            if j == 0:
                return joint_probs
            for a in range(self.N):
                prob_sum += alphas[j][a] * betas[j][a]
            for a in range(self.N):
                joint_probs[a] = (alphas[j][a] * betas[j][a]) / prob_sum
            return joint_probs

        def joint_transition_prob_xj(alphas, betas, j, x):
            '''
            Helper function to calculate P(y^{j}=a, y^j+1=b, x). Returns a 2D
            vector, with the rows iterating over a, and the columns iterating
            over b. j should range from [1, M-1].
            '''
            jt_probs = [[0 for _ in range(self.N)] for _ in range(self.N)]
            if j == 0:
                return jt_probs
            prob_sum = 0
            for a in range(self.N):
                for b in range(self.N):
                    prob_sum += alphas[j][a] * self.O[b][x[j]] * self.A[a][b] * betas[j+1][b]
            for a in range(self.N):
                for b in range(self.N):
                    jt_probs[a][b] =  alphas[j][a] * self.O[b][x[j]] * self.A[a][b] * betas[j+1][b] / prob_sum
            return jt_probs

        for n in range(N_iters):
            print(f'epoch {n+1}/{N_iters}')
            A_nums = np.zeros((self.N, self.N))
            O_nums = np.zeros((self.N, self.V))
            A_denoms = np.zeros((self.N))
            O_denoms = np.zeros((self.N))
            for x in X: # loop over training samples
                M = len(x)
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)
                for j in range(1, M+1): # loop over length of a single sample
                    # Update A
                    joint_probs = joint_prob_xj(alphas, betas, j-1)
                    jt_probs = joint_transition_prob_xj(alphas, betas, j-1, x)
                    for a in range(self.N):
                        A_denoms[a] += joint_probs[a]
                        for b in range(self.N):
                            A_nums[a][b] += jt_probs[a][b]
                    # Update O
                    joint_probs = joint_prob_xj(alphas, betas, j)
                    for a in range(self.N):
                        O_denoms[a] += joint_probs[a]
                        for obs in range(self.V):
                            if obs == x[j-1]:
                                O_nums[a][obs] += joint_probs[a]
            # Update A matrix
            for a in range(self.N):
                denom = A_denoms[a]
                if denom == 0:
                    self.A[a] = np.repeat(1/self.N, self.N)
                else:
                    self.A[a] = A_nums[a] / denom
            # Update O matrix
            for a in range(self.N):
                denom = O_denoms[a]
                if denom == 0:
                    self.O[a] = np.repeat(1/self.V, self.V)
                else:
                    self.O[a] = O_nums[a] / denom
            # Update A_start
            self.A_start = np.zeros((self.N,))
            for x in X:
                self.A_start += self.O.T[x[0]]
            self.A_start /= np.sum(self.A_start)

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the first state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        # (Re-)Initialize random number generator
        rng = np.random.default_rng()

        possible_states = np.arange(0, self.N)
        possible_emissions = np.arange(0, self.V)
        init_state = rng.choice(possible_states, p=self.A_start)

        emission = []
        states = [init_state]

        for _ in range(M):
            prev_state = states[-1]
            emit = rng.choice(possible_emissions, p=self.O[prev_state])
            print(self.A[prev_state])
            state = rng.choice(possible_states, p=self.A[prev_state])
            emission.append(emit)
            states.append(state)    # this will append an extra state at the end
        return emission, states[:-1]


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.N)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    N = len(states)
    V = len(observations)

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(V=V, N=N)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters, seed=None):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''
    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[rng.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[rng.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM