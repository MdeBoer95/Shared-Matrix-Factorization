import numpy as np
import matplotlib.pyplot as plt


def shared_matrix_factorization(D, A, K, steps, alpha=0.002, beta=1, lam=0.02, error_D_ij=[], error_A_ij=[]):
    """
    Applies shared matrix factorization of rank K with SGD to two given matricies D and A.
    @:param D matrix of dimension n x d
    @:param A matrix of dimension n x e
    @:param K rank of the factorization
    @:param steps number of steps for the factorization process with SGD
    @:param alpha stepsize
    @:param beta balancing term for the factorization of A
    @:param lam multiplier for regularization
    @:param error_D_ij error of the factorization of D
    @:param error_A_ij error of the factorization of A
    @:return U, V, H, error_D_ij, error_A_ij    where  U*V' "=" D   and U*H' "=" A
    """

    # Randomly initialize factors U,V and H
    n = D.shape[0]
    d = D.shape[1]
    U = np.random.rand(n, K)
    V = np.random.rand(d, K)

    m = A.shape[1]
    H = np.random.rand(m, K)

    for step in range(steps):

        # calculate and store error for plotting
        e_D = (D - np.dot(U, V.T))
        error_D_ij.append(e_D.sum())
        e_A = (A - np.dot(U, H.T))
        error_A_ij.append(e_A.sum())

        # Choose a matrix to sample form. Either A or D
        choose_sample_matrix = np.random.random_integers(0, 1)

        if choose_sample_matrix == 0:
            # Sample from D
            i, j, sample = sample_entry(D)
            e_ij = e_D[i][j]

            # Update rule for SGD
            for q in range(K):
                U[i][q] = U[i][q] * (1 - alpha*lam/2) + alpha * e_ij * V[j][q]
                V[j][q] = V[j][q] * (1 - alpha*lam) + alpha * e_ij * U[i][q]

        else:
            # Sample from A
            i, p, sample = sample_entry(A)
            e_ip = e_A[i][p]

            # Update rule for SGD
            for q in range(K):
                U[i][q] = U[i][q] * (1 - alpha*lam/2) + alpha * beta * e_ip * H[p][q]
                H[p][q] = H[p][q] * (1 - alpha*lam) + alpha * beta * e_ip * U[i][q]

    return U, V, H, error_D_ij, error_A_ij


def sample_entry(matrix):
    """
    Sample a random entry from a given matrix
    @:param the matrix to sample from
    @:return the sampled entry and its indices
    """
    i = np.random.random_integers(0, matrix.shape[0]-1)
    j = np.random.random_integers(0, matrix.shape[1]-1)

    sample = matrix[i][j]

    return i, j, sample

###############################################################################


if __name__ == "__main__":

    # Data matricies D and A
    D = [
         [5, 3, 0, 1],
         [4, 0, 0, 1],
         [1, 1, 0, 5],
         [1, 0, 0, 4],
         [0, 1, 5, 4],
        ]

    A = [
        [1, 8, 3, 2],
        [1, 2, 0, 1],
        [5, 6, 3, 1],
        [1, 1, 0, 4],
        [0, 1, 0, 6],
        ]
    # convert to np array
    D = np.array(D)
    A = np.array(A)

    steps = 20000

    K = 3

    U, V, H, error_D_ij, error_A_ij = shared_matrix_factorization(D, A, K, steps)

    # Plot error over the number of steps
    plt.plot(np.arange(steps), error_D_ij, label="e_D")
    plt.plot(np.arange(steps), error_A_ij, label="e_A")
    plt.legend()
    plt.title("factorization error over steps")
    plt.ylabel("factorization error")
    plt.xlabel("step")
    plt.show()

    # Show factorization result
    print("original matrix D:", D)
    print("factorization result:",np.dot(U, np.transpose(V)))
    print("\n")
    print("original matrix A:", A)
    print("factorization result:", np.dot(U, np.transpose(H)))