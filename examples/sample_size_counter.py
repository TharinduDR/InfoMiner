import pandas as pd


def sample_size_counter(count_n, count_y, max):

    if count_n > count_y:
        return count_n // count_y if count_n // count_y <= max else max

    else:
        return -(count_y // count_n if count_y // count_n <= max else max)


def data_balancer(train_list):

    shortest_length = 0
    for lang in train_list:
        if shortest_length == 0:
            shortest_length = len(lang)
        elif shortest_length > len(lang):
            shortest_length = len(lang)

    balanced_list = []

    # shuffle
    shuffled_list = train_list[0].sample(frac=1)
    shuffled_list = train_list[1].sample(frac=1)
    shuffled_list = train_list[2].sample(frac=1)

    for i in shortest_length:
        balanced_list.append(train_list[0][i])
        balanced_list.append(train_list[1][i])
        balanced_list.append(train_list[2][i])

    balanced_list = pd.concat(balanced_list)
    return balanced_list
