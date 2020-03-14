import numpy as np
import k_means

# Calculate euclidean distance
def get_distance_matrix(data):
    data_size = data.shape[0]
    result = np.zeros((data_size, data_size))
    for i in range(data_size):
        for j in range(i + 1, data_size):
            result[i][j] = result[j][i] = np.math.sqrt(sum(np.power(data[i] - data[j], 2)))
    return result

# similarity matrix W
def get_W(data, k):
    data_size = data.shape[0]
    dis_m = get_distance_matrix(data[:, 2:])
    W = np.zeros((data_size, data_size))
    for i, j in enumerate(dis_m):
        index_array = np.argsort(j)
        W[i][index_array[1:(k + 1)]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W + W) / 2
    return W

# degree matrix D
def get_D(W):
    return np.diag(sum(W))

# laplacian matrix L
def get_L(W, D):
    return (D - W)

# feature matrix H
def get_H(L, k):
    eigval, eigvec = np.linalg.eig(L)
    ix = np.argsort(eigval)[0:k]
    return eigvec[:, ix]

def start(data, k):
    W = get_W(data, k)
    D = get_D(W)
    L = get_L(D, W)
    H = get_H(L, k)
    result = k_means.start(H, k)
    return result