import numpy as np
from cProfile import Profile
from pstats import SortKey, Stats
from byte_pair_encoding import bpe_generator

corpus = []
for file_name in ["mnist_data.py", "mnist.py", "byte_pair_encoding.py", "use_model.py"]:
    with open(file_name, "r") as f:
        corpus.append(f.read())

corpus = 100*corpus

with Profile() as profile:
    print(bpe_generator(corpus, 128))
    (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.TIME)
        .print_stats()
    )