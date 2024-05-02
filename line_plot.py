# x unit: 0, 1, 2, 3 qry expansion
# x: support expansion
# x unit row: animals mix-up, animals image translator, 
# cub mix-up, miniImagenet mix-up
import matplotlib.pyplot as plt

# 0 query
animals_mixup0 = [54.64, 53.76, 52.08, 50.64]
animals_trans0 = [54.64, 55.32, 54.12, 55.44]
cub_mixup0 = [44.61, 44.76, 45.28, 44.04]
mini_mixup0 = [55.47, 53.80, 53.48, 51.12]
qry0 = [animals_mixup0, animals_trans0, cub_mixup0, mini_mixup0]

# 1 query
animals_mixup1 = [75.80, 94.04, 92.20, 90.84]
animals_trans1 = [74.72, 96.00, 94.72, 92.96]
cub_mixup1 = [80.48, 95.52, 94.64, 93.84]
mini_mixup1 = [83.28, 99.36, 97.80, 97.40]
qry1 = [animals_mixup1, animals_trans1, cub_mixup1, mini_mixup1]

# 2 query
animals_mixup2 = [80.48, 96.72, 99.92, 99.40]
animals_trans2 = [82.36, 98.20, 99.56, 99.80]
cub_mixup2 = [87.84, 98.76, 99.84, 99.72]
mini_mixup2 = [90.00, 99.48, 99.96, 99.96]
qry2 = [animals_mixup2, animals_trans2, cub_mixup2, mini_mixup2]

# 3 query
animals_mixup3 = [81.96, 97.68, 99.84, 100.00]
animals_trans3 = [84.40, 98.92, 99.84, 100.00]
cub_mixup3 = [90.56, 99.12, 99.84, 100.00]
mini_mixup3 = [91.80, 99.48, 100.00, 100.00]
qry3 = [animals_mixup3, animals_trans3, cub_mixup3, mini_mixup3]

stats = [qry0, qry1, qry2, qry3]

plt.rcParams['font.family'] = 'Times New Roman'

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
for qry_expansion_i in range(4):
    data_mixup, data_trans, cub_mixup, mini_mixup = stats[qry_expansion_i]
    axes[qry_expansion_i].plot(data_mixup, label='Animals Mix-Up', marker='o')
    axes[qry_expansion_i].plot(data_trans, label='Animals Trans', marker='o') 
    axes[qry_expansion_i].plot(cub_mixup, label='CUB Mix-Up', marker='o') 
    axes[qry_expansion_i].plot(mini_mixup, label='Mini Mix-Up', marker='o') 
    axes[qry_expansion_i].set_xlabel('Extra Support', fontsize=14)
    axes[qry_expansion_i].set_ylabel('Acc', fontsize=14)
    axes[qry_expansion_i].set_title(f'{qry_expansion_i} Extra Query', fontsize=16)
    axes[qry_expansion_i].grid(True)
plt.tight_layout()
plt.legend(loc='lower right', ncol=5, fontsize=15)
plt.savefig('lineplot.pdf')