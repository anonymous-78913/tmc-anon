import sys

Ks = [1, 3, 10, 30, 100, 300, 1000]
seeds = range(50)
result_list = ["non-fac/{}_{:05d}_{:02d}.row".format(fc_nf, K, seed) for fc_nf in ["fc", "nf"] for K in Ks for seed in seeds]

result_str = " ".join(result_list)
print(result_str)
