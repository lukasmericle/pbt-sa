from numpy import array, reshape

def read_file(filename):
    """Read file provided by researchers.
    https://www.researchgate.net/publication/271198281_Benchmark_instances_for_the_Multidimensional_Knapsack_Problem"""
    with open(filename, "r") as f:
        lines = f.readlines()
    splitlines = [line.split() for line in lines]
    all_numbers = list(map(int, [n for line in splitlines for n in line]))

    n = all_numbers[0]
    m = all_numbers[1]
    v = all_numbers[2]
    all_numbers = all_numbers[3:]

    item_values = array(all_numbers[:n])
    all_numbers = all_numbers[n:]

    item_weights = reshape(all_numbers[:m*n], (m,n)).T
    all_numbers = all_numbers[m*n:]

    knapsack_capacities = array(all_numbers)

    return item_values, item_weights, knapsack_capacities, v
