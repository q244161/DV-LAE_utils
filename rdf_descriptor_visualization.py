import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 定义第一个脚本中的截断函数
def cutoff_function(r, rc):
    x = r/rc
    return (np.tanh(1-x))**3
# 生成距离数组
r_ij = np.linspace(0, 6, 500)

# 读取第二个脚本的RDF数据
file_path = 'rdf_Mg_Mg.out'

# 加载数据
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=14, header=None)
dist_bin_l = data[0]
rdf = data[3]

# 绘图
plt.figure(figsize=(12, 8))

# 第一个脚本的绘图命令
params = [
    {'eta': 1.5, 'r_s': 5.5, 'rc': 10.0},
    {'eta': 4.0, 'r_s': 3.5, 'rc': 7.0},
    {'eta': 2.0, 'r_s': 4.0, 'rc': 8.0},
    {'eta': 2.0, 'r_s': 4.5, 'rc': 8.0},
    {'eta': 2.0, 'r_s': 5.0, 'rc': 8.0},
    {'eta': 3.5, 'r_s': 3.5, 'rc': 9.0},
]

for p in params:
    eta, r_s, rc = p['eta'], p['r_s'], p['rc']
    G2_i = np.exp(-eta * (r_ij - r_s)**2) * [cutoff_function(r, rc) for r in r_ij]
    plt.plot(r_ij, G2_i, label=f'η={eta}, r_s={r_s}, rc={rc}')

# 第二个脚本的绘图命令
plt.plot(dist_bin_l, rdf, label='RDF of Mg-Mg')

# 设置图例和其他绘图属性
plt.xlabel('Distance (Å)', fontsize=24)
plt.ylabel('Value', fontsize=24)
plt.title('Combined Plot of Symmetry Functions and RDF', fontsize=20)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()
