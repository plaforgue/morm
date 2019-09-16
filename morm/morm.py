import numpy as np
from numba import njit
from sklearn.metrics.pairwise import euclidean_distances


#####################
#                   #
#   MoM Estimator   #
#                   #
#####################

def partition_blocks(X, K):
    """Partition X into K disjoint blocks as large as possible
    """
    # get largest block size (should be at least 1)
    n = len(X)
    B = n // K
    if B == 0:
        raise ValueError("Invalid number of blocks %s, "
                         "larger than number of samples %s" % (K, n))

    # create and return blocks (plus block size)
    np.random.shuffle(X)
    X = X[:K * B]
    blocks = np.array(np.split(X, K))

    return blocks, B


def MoM(X, K):
    """Compute mom estimate with K blocks
    """
    blocks, _ = partition_blocks(X, K)
    means = np.mean(blocks, axis=1)
    mom = np.median(means)

    return mom


#######################
#                     #
#   MoRM Estimators   #
#                     #
#######################

@njit
def swor(n, p):
    """Efficient SWoR
    """
    idx = np.zeros(p, dtype=np.int32)
    mask = np.ones(n, dtype=np.bool_)
    count = 0

    while count < p:
        i = np.random.randint(low=0, high=n)
        if mask[i]:
            mask[i] = 0
            idx[count] = i
            count += 1

    return idx


def swor_blocks(X, K, B):
    """Sample K SWoR blocks of size B from X
    """
    n, d = X.shape
    blocks = np.zeros((K, B, d))

    for k in range(K):
        idx = swor(n, B)
        blocks[k, :, :] = X[idx, :]

    return blocks


@njit
def mc(n, p):
    """Efficient MC
    """
    idx = np.zeros(p, dtype=np.int32)
    for i in range(p):
        idx[i] = np.random.randint(low=0, high=n)
    return idx


def mc_blocks(X, K, B):
    """Sample K MC blocks of size B from X
    """
    n, d = X.shape
    blocks = np.zeros((K, B, d))

    for k in range(K):
        idx = mc(n, B)
        blocks[k, :, :] = X[idx, :]

    return blocks


def MoRM(X, K, B, sampling='SWoR'):
    """Compute morm estimate with K blocks of size B
    """
    if sampling == 'SWoR':
        blocks = swor_blocks(X, K, B)
    elif sampling == 'MC':
        blocks = mc_blocks(X, K, B)

    means = np.mean(blocks, axis=1)
    morm = np.median(means)

    return morm


###############################
#                             #
#   Mo(I)U-stats Estimators   #
#                             #
###############################

def u_mat(X, kernel='squared_norm'):
    """Compute comparison matrix from specified kernel
       To modify in order to allow for more kernels
    """
    if kernel == 'squared_norm':
        M = euclidean_distances(X)
        M **= 2
        M /= 2

    # elif kernel == 'my_kernel':
    #     M = my_kernel(X)

    return M


def ustat_c(X, kernel='squared_norm'):
    """Compute complete Ustat from specified kernel
    """
    n = len(X)
    M = u_mat(X, kernel=kernel)
    U = M.sum() - np.trace(M)
    U /= n * (n - 1)
    return U


def MoCU(X, K, kernel='squared_norm', sampling='partition', B=10):
    """Compute mocu estimate from specified kernel
    """
    # sample blocks
    if sampling == 'partition':
        blocks, B = partition_blocks(X, K)
        if B < 2:
            raise ValueError("Invalid number of blocks %s, "
                             "less than 2 samples per block")
    elif sampling == 'SWoR':
        blocks = swor_blocks(X, K, B)
    elif sampling == 'MC':
        blocks = mc_blocks(X, K, B)

    # compute complete ustats and return mocu
    Ustats = np.zeros(K)
    for k, block in enumerate(blocks):
        Ustats[k] = ustat_c(block, kernel=kernel)
    mocu = np.median(Ustats)

    return mocu


def u_vec(X1, X2, kernel='squared_norm'):
    """Compute comparison vector from specified kernel
       To modify in order to allow for more kernels
    """
    if kernel == 'squared_norm':
        Y = X1 - X2
        v = np.linalg.norm(Y, axis=1)
        v **= 2
        v /= 2

    # elif kernel == 'my_kernel':
    #     v = my_kernel(X1, X2)

    return v


def ustat_i(X1, X2, kernel='squared_norm'):
    """Compute incomplete Ustat based on X1 and X2 and specified kernel
    """
    v = u_vec(X1, X2, kernel=kernel)
    U = np.mean(v)
    return U


def pix_to_dix(k, n):
    """Transform an index on the pairs (from 0 to n(n-1) - 1) into the double
       index of the corresponding pair (with symmetry)
    """
    # standard transformation
    i = k // (n - 1)
    j = k % (n - 1)
    # to remove diagonal
    c0 = i < n - 1
    c1 = j >= i
    j += c0 * c1
    return i, j


def MoIU(X, K, B, kernel='squared_norm', sampling='SWoR'):
    """Compute moiu estimate from specified kernel
    """
    n = len(X)
    Ustats = np.zeros(K)

    for k in range(K):
        if sampling == 'SWoR':
            p_idxs = swor(n * (n - 1), B)
        elif sampling == 'MC':
            p_idxs = mc(n * (n - 1), B)

        d_idxs = np.array(pix_to_dix(p_idxs, n))

        X1 = X[d_idxs[0, :]]
        X2 = X[d_idxs[1, :]]

        Ustats[k] = ustat_i(X1, X2)

    moiu = np.median(Ustats)

    return moiu
