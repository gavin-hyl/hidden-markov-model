from hmm import HiddenMarkovModel

def index_sequence(seqs: list[list[object]]):
    '''
    Indexes a list of sequences.

    Arguments:
        seqs:   a list of sequences

    Returns:
        A tuple containing the indexed list and the object to index mapping
    '''
    obj_to_idx = {}
    idx = 0
    for seq in seqs:
        for s in seq:
            if obj_to_idx.get(s) is None:
                obj_to_idx.update({s: idx})
                idx += 1
    indexed = []
    for seq in seqs:
        indexed.append([obj_to_idx.get(s) for s in seq])
    return indexed, obj_to_idx

def invert_map(mapping: dict):
    '''
    Inverts a dictionary mapping.

    Arguments:
        mapping:    the dictionary to invert
    '''
    return {v:k for k, v in mapping.items()}

def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of objects.
        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of objects.
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
    X_indexed, X_oi = index_sequence(X)
    Y_indexed, Y_oi = index_sequence(Y)
    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(V=V, N=N, X_obj_to_idx=X_oi, Y_obj_to_idx=Y_oi)
    HMM.supervised_learning(X_indexed, Y_indexed)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
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
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    V = len(observations)
    X_indexed, X_oi = index_sequence(X)

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(V=V, N=n_states, X_obj_to_idx=X_oi)
    HMM.unsupervised_learning(X_indexed, N_iters)

    return HMM