import itertools

base_letters = ["A", "T", "G", "C"]
all_letters = base_letters + ["N"]
combinations = list(itertools.product(all_letters, repeat=3))
combinations_with_N = ["".join(c) for c in combinations if "N" in c]

print (combinations_with_N)