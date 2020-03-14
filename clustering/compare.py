import numpy as np

# Judge two documents of the same cluster are similar or not
def judge_same_cho(index_1, index_2):
    result = False
    if (index_1 > 331 and index_2 > 331):
        result = True
    elif (index_1 > 277 and index_2 > 277):
        result = True
    elif (index_1 > 202 and index_2 > 202):
        result = True
    elif (index_1 > 67 and index_2 > 67):
        result = True
    elif (index_1 > 0 and index_2 > 0):
        result = True
    return result

def judge_same_iyer(index_1, index_2):
    result = False
    if (index_1 > 491 and index_2 > 491):
        result = True
    elif (index_1 > 472 and index_2 > 472):
        result = True
    elif (index_1 > 409 and index_2 > 409):
        result = True
    elif (index_1 > 392 and index_2 > 392):
        result = True
    elif (index_1 > 354 and index_2 > 354):
        result = True
    elif (index_1 > 342 and index_2 > 342):
        result = True
    elif (index_1 > 299 and index_2 > 299):
        result = True
    elif (index_1 > 261 and index_2 > 261):
        result = True
    elif (index_1 > 100 and index_2 > 100):
        result = True
    elif (index_1 > 0 and index_2 > 0):
        result = True
    return result

def find(start, end, list):
    total = 0
    for i in range(start, end):
        if (i not in list):
            total = total + 1
    return total

# Calculate how many similar document pairs are in different clusters
def calc_fn_cho(list, num):
    result = 0
    if (num > 331):
        result = find(331, 386, list)
    elif (num > 277):
        result = find(277, 330, list)
    elif (num > 202):
        result = find(202, 276, list)
    elif (num > 67):
        result = find(67, 201, list)
    elif (num > 0):
        result = find(0, 66, list)
    return result

def calc_fn_iyer(list, num):
    result = 0
    if (num > 491):
        result = find(492, 516, list)
    elif (num > 472):
        result = find(473, 491, list)
    elif (num > 409):
        result = find(410, 472, list)
    elif (num > 392):
        result = find(393, 406, list)
    elif (num > 354):
        result = find(355, 388, list)
    elif (num > 342):
        result = find(343, 349, list)
    elif (num > 299):
        result = find(300, 342, list)
    elif (num > 261):
        result = find(262, 295, list)
    elif (num > 100):
        result = find(101, 245, list)
    elif (num > 0):
        result = find(1, 100, list)
    return result

# Precision, Recall and F-score
def external(name, result, k):
    TP = 0 # two similar documents to the same cluster
    FP = 0 # two dissimilar documents to the same cluster
    FN = 0 # two similar documents to different clusters
    for c in range(k):
        temp = result[c]
        for i in range(len(temp)):
            for j in range((i + 1), len(temp)):
                if (name == "cho"):
                    same = judge_same_cho(temp[i], temp[j])
                else:
                    same = judge_same_iyer(temp[i], temp[j])
                if (same == True):
                    TP = TP + 1
                else:
                    FP = FP + 1
            if (name == "cho"):
                FN = FN + calc_fn_cho(temp, temp[i])
            else:
                FN = FN + calc_fn_iyer(temp, temp[i])

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)
    print("Precision is: " + str(P))
    print("Recall is: " + str(R))
    print("F-score is: " + str(F1))

# Calculate value of center point
def compute_center(total):
    total_num = len(total)
    if (total_num == 0):
        return
    else:
        data_size = len(total[0])
        center = []
        for i in range(data_size):
            sum = 0
            for j in range(total_num):
                sum  = sum + total[j][i]
            center.append(sum / total_num)
        return center

# Calculate euclidean distance
def calc_euclidean(list_1, list_2):
    result = sum(np.power(list_1 - list_2, 2))
    return result

# sum of squared error (SSE)
def internal(data, result, k):
    sse = 0
    for c in range(k):
        temp_1 = result[c]
        total = []
        for i in range(len(temp_1)):
            total.append(data[temp_1[i]])

        center = compute_center(total)

        for i in range(len(total)):
            temp_2 = total[i]
            sse = sse + calc_euclidean(np.array(temp_2[2:]), np.array(center[2:]))

    print("SSE is: " + str(sse))
