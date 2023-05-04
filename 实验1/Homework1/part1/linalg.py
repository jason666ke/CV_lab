import numpy as np
import numpy.linalg


def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    When inputs are 2-D array, `np.matmul()` and `np.dot()` have same result, 
    you can also use `np.matmul()`.

    notice that `np.dot()` and `np.matmul()` need `a` with shape (x, n), `b` with shape `(n, x)
    so you need to transpose `a`, you can use syntax `a.T`.


    Args:
        a: numpy array of shape (n, x)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """
    out = np.dot(a.T, b)
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a^Tb) x (Ma), `a^T` is transpose of `a`, 
    (a^Tb) is matrix multiplication of a^T and b,
    (Ma) is matrix multiplication of M and a.

    You can use `np.matmul()` to do matrix multiplication.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (n, 1).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    left = dot_product(a, b)
    right = np.dot(M, a)
    if left.shape == (1, 1):
        out = left[0, 0] * right
    else:
        out = np.dot(left, right)
    return out


def eigen_decomp(M):
    """Implement eigenvalue decomposition.

    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, m)

    Returns:
        w: numpy array of shape (m,) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w = None
    v = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return w, v


def euclidean_distance_native(u, v):
    """Computes the Euclidean distance between two vectors, represented as Python
    lists.

    Args:
        u (List[float]): A vector, represented as a list of floats.
        v (List[float]): A vector, represented as a list of floats.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    assert isinstance(u, list)
    assert isinstance(v, list)
    assert len(u) == len(v)

    # Compute the distance!
    # Notes:
    #  1) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.

    ### YOUR CODE HERE
    square_sum = 0
    for index in range(len(u)):
        square_sum += np.power(u[index]-v[index], 2)
    return np.sqrt(square_sum)
    ### END YOUR CODE


def euclidean_distance_numpy(u, v):
    """Computes the Euclidean distance between two vectors, represented as NumPy
    arrays.

    Args:
        u (np.ndarray): A vector, represented as a NumPy array.
        v (np.ndarray): A vector, represented as a NumPy array.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert u.shape == v.shape

    # Compute the distance!
    # Note:
    #  1) You shouldn't need any loops
    #  2) Some functions you can Google that might be useful:
    #         np.sqrt(), np.sum()
    #  3) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.

    ### YOUR CODE HERE
    return np.linalg.norm(u - v)
    ### END YOUR CODE


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    c = np.linalg.eig(M)
    eigen_map = []
    for index in range(len(c[0])):
        eigen_map.append([c[0][index], c[1][index]])
    # 按照特征值大小排序
    eigen_map.sort(key=lambda x: x[0])
    # 输出结果
    eigenvalues = []
    eigenvectors = []
    for index in range(k):
        reverse_index = -(index + 1)
        eigenvalues.append(eigen_map[reverse_index][0])
        eigenvectors.append(eigen_map[reverse_index][1])

    return eigenvalues, eigenvectors
