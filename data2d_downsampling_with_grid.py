import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def huafenwangge(data_2d, interval, inputpath, mode, max_points_per_grid=1, output=None, plotflag=False):
    a = np.load(data_2d)
    # 设置网格间隔和每个网格中最多取的数据点数
    # 获取数据的范围
    x_min = np.min(a[:, 0])  # 假设 x 数据在第一列
    x_max = np.max(a[:, 0])
    y_min = np.min(a[:, 1])  # 假设 y 数据在第二列
    y_max = np.max(a[:, 1])

    # 根据间隔划分网格
    x_bins = np.arange(x_min, x_max + interval, interval)
    y_bins = np.arange(y_min, y_max + interval, interval)
    # print(len(x_bins), len(y_bins))
    # 划分网格并取数据
    grid_data = []

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            x_min_bin = x_bins[i]
            x_max_bin = x_bins[i + 1]
            y_min_bin = y_bins[j]
            y_max_bin = y_bins[j + 1]

            # 获取每个网格内的数据索引
            indices_in_grid = np.where(
                (a[:, 0] >= x_min_bin) & (a[:, 0] < x_max_bin) &
                (a[:, 1] >= y_min_bin) & (a[:, 1] < y_max_bin)
            )[0]

            # 根据需要采取规则
            if len(indices_in_grid) > 0:
                # 获取最多指定数量的数据点的索引
                if len(indices_in_grid) > max_points_per_grid:
                    np.random.shuffle(indices_in_grid)  # 随机打乱顺序
                    indices_in_grid = indices_in_grid[:max_points_per_grid]

                # 保存单个网格中的索引到整体列表中
                grid_data.append(indices_in_grid.tolist())  # 转换为 Python 列表并添加

    # 将 grid_data 转换为平铺的单个索引列表
    flat_indices = np.concatenate(grid_data)

    plotscatter(plotflag, a, flat_indices, x_bins, y_bins)

    if output is None:
        path_info = os.path.split(data_2d)
        directory = path_info[0]  # 文件所在目录
        file_name = path_info[1]  # 文件名

        # 从文件名中分割出指定部分
        file_name_parts = file_name.split('_')
        part_1 = file_name_parts[3]  # 获取 '202401081123'
        part_2 = '_'.join(file_name_parts[4:7])  # 获取 'trainf_5_tsne'
        # part_3 = file_name_parts[-1].split('.')[0]  # 获取 '2'（去除扩展名）
        name2 = f'output_{part_1}_{part_2}_{interval}_{max_points_per_grid}_{len(flat_indices)}.data'
        output = os.path.join(directory, name2)
    else:
        path_info = os.path.split(data_2d)
        directory = rf'mode{mode}'
        file_name = path_info[1]  # 文件名

        # 从文件名中分割出指定部分
        file_name_parts = file_name.split('_')
        part_1 = file_name_parts[3]  # 获取 '202401081123'
        part_2 = '_'.join(file_name_parts[4:7])  # 获取 'trainf_5_tsne'
        # part_3 = file_name_parts[-1].split('.')[0]  # 获取 '2'（去除扩展名）
        name2 = f'output_{part_1}_{part_2}_{interval}_{max_points_per_grid}_{len(flat_indices)}.data'
        output = os.path.join(directory , name2)
    print('筛选前数量：', len(a))
    saveoutput(flat_indices, inputpath, output)


# 绘制根据索引筛选出的特定散点图
def plotscatter(plotflag, a, flat_indices, x_bins, y_bins):
    if plotflag:
        # 根据索引绘制筛选出的散点图
        selected_points = a[flat_indices]  # 根据索引筛选出对应的点
        # 绘制所有散点图
        # plt.scatter(a[:, 0], a[:, 1], color='blue', label='All Points')
        plt.scatter(selected_points[:, 0], selected_points[:, 1], color='red', label='Selected Points')
        # 绘制网格
        for x in x_bins:
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

        for y in y_bins:
            plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
        # 设置图例和标题等其他绘图参数
        plt.legend()
        plt.title('Scatter Plot with Selected Points')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # 显示图形
        plt.show()


def saveoutput(flat_indices, inputpath, output):
    indexarr = flat_indices
    begin_found = False
    index = 0
    file_extension = os.path.splitext(inputpath)[1]
    if file_extension == '.data':
        # 使用自定义函数read_n2p2读取data文件
        with open(inputpath, 'r', newline='\n') as input_f, open(output, 'w', newline='\n') as output_f:
            for line in input_f:
                if begin_found:
                    if line.strip() == 'end':
                        begin_found = False
                        if index in indexarr:
                            output_f.write(line)
                        index += 1
                    else:
                        if index in indexarr:
                            output_f.write(line)
                elif line.strip() == 'begin':
                    if index in indexarr:
                        output_f.write(line)
                    begin_found = True
    else:
        # 使用ASE读取其他格式文件
        from ase.io import read,write
        atomslist = read(inputpath, index=':')
        selected_atoms = [atomslist[i] for i in flat_indices]
        write(output, selected_atoms, format='xyz')



    print('筛选后数量：', len(flat_indices))
    print('输出数据:', output)


def find_matching_files(base_path, prefix, numbers, mode):
    matching_files = []
    for number in numbers:
        file_pattern = os.path.join(base_path, f'{prefix}_*_{number}_tsne_{mode}.npy')
        matching_files.extend(glob.glob(file_pattern))
    return matching_files


if __name__ == '__main__':
    inputpath = r'input.data'
    base_path = r'.'
    prefix = 'data_2d_b'

    # 指定的数字范围
    numbers = [40]
    modes = [2]

    interval = 0.1  # 按需修改间隔大小
    max_points_per_grid = 1  # 每个网格中最多取的数据点数
    
    outbase = r'.\output'
    for mode in modes:
        matching_files = find_matching_files(base_path, prefix, numbers, mode)
        # output= os.path.join(outbase, f'mode{mode}')
        output = rf'.\output\mode{mode}'
        for data_2d in matching_files:
            print('*' * 50)
            print(data_2d)
            huafenwangge(data_2d, interval, inputpath, mode, max_points_per_grid=max_points_per_grid, output=output,
                         plotflag=False)
            # print('*' * 50)
