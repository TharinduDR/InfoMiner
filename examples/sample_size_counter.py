def sample_size_counter(count_n, count_y, max):

    if count_n > count_y:
        return count_n // count_y if count_n // count_y <= max else max

    else:
        return -(count_y // count_n if count_y // count_n <= max else max)

