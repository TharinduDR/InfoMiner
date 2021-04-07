def sample_size_counter(count_n, count_y):

    if count_n > count_y:
        return count_n // count_y if count_n // count_y <= 2 else 2

    else:
        return -(count_y // count_n if count_y // count_n <= 2 else 2)

