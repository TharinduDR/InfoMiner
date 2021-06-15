import pandas as pd


def sample_size_counter(count_n, count_y, max):

    if count_n > count_y:
        return count_n // count_y if count_n // count_y <= max else max

    else:
        return -(count_y // count_n if count_y // count_n <= max else max)


def data_balancer(unbalanced_list):

    shortest_length = 0
    for lang in unbalanced_list:
        if shortest_length == 0:
            shortest_length = len(lang)
        elif shortest_length > len(lang):
            shortest_length = len(lang)

    print("shortest_length: ", shortest_length)
    balanced_list = []

    # shuffle
    shuffled_list = unbalanced_list[0].sample(frac=1)
    shuffled_list = unbalanced_list[1].sample(frac=1)
    shuffled_list = unbalanced_list[2].sample(frac=1)

    for i in range(shortest_length):
        balanced_list.append(unbalanced_list[0].iloc[[i]])
        balanced_list.append(unbalanced_list[1].iloc[[i]])
        balanced_list.append(unbalanced_list[2].iloc[[i]])

    balanced_list = pd.concat(balanced_list)
    print(balanced_list.head())

    return balanced_list
