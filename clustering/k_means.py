import numpy as np

# Select k centroids randomly
def get_centroid(data, k):
    attribute_size = data.shape[1]
    centroids = np.zeros((k, attribute_size))
    for i in range(2, attribute_size):
        # Find the min/max value in each column of attributes
        min_value = float(min(data[:, i]))
        max_value = float(max(data[:, i]))
        # Select centroids in the interval randomly
        value_range = max_value - min_value
        centroids[:, i] = (min_value + value_range * np.random.rand(k, 1)).flatten()
    return centroids

# Calculate euclidean distance
def calc_euclidean(list_1, list_2):
    result = np.math.sqrt(sum(np.power(list_1 - list_2, 2)))
    return result

def start(data, k):
    data_size = data.shape[0]
    centroids = get_centroid(data, k)
    # Index & Distance
    cluster_assment = np.full((data_size, 2), -1.0)

    flag = True

    save_list = []
    for i in range(k):
        save_list.append(0)

    while(flag):
        flag = False
        # Classify the data point into the centroid with the shortest distance
        for i in range(data_size):
            min_distance = np.inf
            min_index = -1
            # Find the closest centroid
            for j in range(k):
                list_1 = centroids[j, 2:]
                list_2 = data[i, 2:]
                distance = calc_euclidean(list_1, list_2)
                if (distance < min_distance):
                    min_distance = round(distance, 2)
                    min_index = j
            if (cluster_assment[i, 0] != min_index):
                cluster_assment[i, 0] = min_index
                cluster_assment[i, 1] = min_distance
                flag = True

        # Update the centroid
        # Use the mean of the points in each class
        for i in range(k):
            index = cluster_assment[:, 0]
            # Find all points in class-i
            point_index_list = []
            for j in range(index.size):
                if (index[j] == i):
                    point_index_list.append(j)
            save_list[i] = point_index_list
            point_list = []
            for j in range(len(point_index_list)):
                temp = point_index_list[j]
                point_list.append(data[temp])
            # Calculate the mean
            if (len(point_list) != 0):
                centroids[i, :] = np.mean(point_list, axis=0)

    return save_list