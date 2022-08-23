import numpy as np

#auxiliary function
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # Abnormal situation
        raise Exception("input {0} is not in allowable_set{1}:".format(
            x,allowable_set))
    # lambda function: (==)if equal return True else False
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    #map(function,list)---return new_list[f->list[i]]
    return list(map(lambda s: x == s, allowable_set))

# def get_len_matrix(len_list):
#     len_list = np.array(len_list)
#     max_nodes = np.sum(len_list)
#     curr_sum = 0
#     len_matrix = []
#     for l in len_list:
#         curr = np.zeros(max_nodes)
#         curr[curr_sum:curr_sum + l] = 1
#         len_matrix.append(curr)
#         curr_sum += l
#     return np.array(len_matrix)


def get_len_matrix(len_list):
    len_list = np.array(len_list)
    max_nodes = np.sum(len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum + l] = 1
        len_matrix.append(curr)
        curr_sum += l
    return np.array(len_matrix)

