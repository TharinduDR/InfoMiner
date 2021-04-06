def sample_size_counter(count_n, count_y):

    if count_n > count_y:
        return count_n // count_y if count_n // count_y <= 3 else 3

    else:
        return -(count_y // count_n if count_y // count_n <= 3 else 3)

