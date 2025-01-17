import os
import subprocess
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime
from functools import partial

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.visualize import view
from matplotlib.widgets import Button
from tqdm import tqdm
import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.offline import plot
from ase.io import write
import tkinter as tk
from tkinter import messagebox
import glob
import time


global flag1
flag1 = False


class Molecule:
    def __init__(self, num_atoms):
        self.num_atoms = num_atoms
        self.atoms = []
        self.index = None

    def add_atom(self, atom_number, g_value):
        self.atoms.append((atom_number, g_value))

    def get_atomic_number_vector(self):
        atomic_number_vector = []
        for atom_number, _ in self.atoms:
            atomic_number_vector.append(atom_number)
        return atomic_number_vector

    def get_atomic_element(self):
        atomic_number_vector = []
        for atom_number, _ in self.atoms:
            atomic_number_vector.append(atom_number)
        return list(set(atomic_number_vector))

    def get_feature_vector(self):
        feature_vector = []
        for _, g_value in self.atoms:
            feature_vector.extend(g_value)
        return feature_vector

    def get_feature_matrix(self):
        feature_matrix = []
        for element, g_value in self.atoms:
            feature_matrix.append(g_value)
        return feature_matrix

    def get_feature_matrix_by_element(self, ele):
        feature_matrix = []
        for element, g_value in self.atoms:
            if ele == element:
                feature_matrix.append(g_value)
        return np.array(feature_matrix)


def read_molecules(filename):
    molecule_dict = []  # 创建一个空字典来存储分子
    index = 0
    with open(filename, 'r') as file:
        while True:
            line = file.readline().strip()
            if not line:
                return molecule_dict

            if line.startswith('#'):  # 忽略以 '#' 开头的行
                continue

            num_atoms = int(line)
            molecule = Molecule(num_atoms)
            molecule.index = index
            index = index + 1
            for _ in range(num_atoms):
                atom_info = file.readline().strip().split()
                atom_number = int(atom_info[0])
                g_value = list(map(float, atom_info[1:]))
                molecule.add_atom(atom_number, g_value)

            # Skip an additional line
            file.readline()

            molecule_dict.append(molecule)  # 如果feature_vector已存在，将当前分子添加到对应的列表中
            # if len(molecule_dict) > 10:
            #     return molecule_dict

    return molecule_dict


def read_molecule(filename, index=1):
    if index == 0:
        index = 99999999
    index1 = 0
    if index >= 0:
        with open(filename, 'r', encoding='utf-8') as file:
            while True:
                line = file.readline().strip()
                if index - 1 == index1:
                    if not line or line == '\x1a':
                        return index1
                    if line.startswith('#'):  # 忽略以 '#' 开头的行
                        continue

                    num_atoms = int(line)
                    molecule = Molecule(num_atoms)
                    molecule.index1 = index1
                    for _ in range(num_atoms):
                        atom_info = file.readline().strip().split()
                        atom_number = int(atom_info[0])
                        g_value = list(map(float, atom_info[1:]))
                        molecule.add_atom(atom_number, g_value)
                    file.readline()
                    return molecule  # 如果feature_vector已存在，将当前分子添加到对应的列表中
                else:
                    if not line or line == '\x1a':
                        return index1
                    if line.startswith('#'):  # 忽略以 '#' 开头的行
                        continue
                    num_atoms = int(line)
                    index1 = index1 + 1
                    for _ in range(num_atoms):
                        file.readline()

                    if filename.endswith('.data'):
                        file.readline()
    else:
        last_n_structures = []  # 用于存储最后n个结构的列表
        with open(filename, 'r', encoding='utf-8') as file:
            while True:
                line = file.readline().strip()
                if not line or line == '\x1a':
                    return last_n_structures[0], index1
                if line.startswith('#'):  # 忽略以 '#' 开头的行
                    continue
                num_atoms = int(line)
                index1 = index1 + 1
                molecule = Molecule(num_atoms)
                molecule.index1 = index1
                for _ in range(num_atoms):
                    atom_info = file.readline().strip().split()
                    atom_number = int(atom_info[0])
                    g_value = list(map(float, atom_info[1:]))
                    molecule.add_atom(atom_number, g_value)
                file.readline()

                # 维护一个长度为n的滑动窗口，记录最后n个结构
                if len(last_n_structures) == abs(index):
                    last_n_structures.pop(0)  # 删除最早的结构，保持滑动窗口长度为n
                last_n_structures.append(molecule)


def functionname1(molecule, maxvalue, minvalue, intevellist, elementlist, intervelnum):
    elements = molecule.get_atomic_element()
    gdic = {}
    for ele in elementlist:
        if ele in elements:
            matrix_by_element = (molecule.get_feature_matrix_by_element(ele)).T
            gdic[ele] = []
            for i, e in enumerate(matrix_by_element):
                # 计算整数区间的边界，包含最大最小值
                if intevellist[ele][i] != 0:
                    hist, bin_edges = np.histogram(e,
                                                   bins=np.arange(minvalue[ele][i],
                                                                  maxvalue[ele][i] + intevellist[ele][i],
                                                                  intevellist[ele][i]))
                else:
                    hist = np.array([len(e)] + [0] * (intervelnum - 1))
                    bin_edges = np.arange(np.unique(e)[0], np.unique(e)[0] + 15, 1)
                gdic[ele].append((hist, bin_edges))
        else:
            gdic[ele] = []
            # 处理当分子中没有该元素时的情况
            for i in range(len(intevellist[ele])):
                # 使用空数组来计算直方图，因为没有该元素的特征
                if intevellist[ele][i] != 0:
                    hist, bin_edges = np.histogram(np.array([]),
                                                   bins=np.arange(minvalue[ele][i],
                                                                  maxvalue[ele][i] + intevellist[ele][i],
                                                                  intevellist[ele][i]))
                else:
                    hist = np.array([0] + [0] * (intervelnum - 1))  # 没有数据的直方图
                    bin_edges = np.arange(minvalue[ele][i], minvalue[ele][i] + 15, 1)  # 根据需要调整bin的边界
                gdic[ele].append((hist, bin_edges))
            # gdic[ele] = []
            # for i in range(len(intevellist[ele])):
            #     if intevellist[ele][i] != 0:
            #         hist, bin_edges = np.histogram(np.array([]),
            #                                        bins=np.arange(minvalue[ele][i],
            #                                                       maxvalue[ele][i] + intevellist[ele][i],
            #                                                       intevellist[ele][i]))
            #     else:
            #         hist = np.array([len(e)] + [0] * (intervelnum - 1))
            #         bin_edges = np.arange(np.unique(e)[0], np.unique(e)[0] + 15, 1)
            #     gdic[ele].append((hist, bin_edges))
            pass

    return gdic


def read_n2p2(filename='output.data', index=':', with_energy_and_forces='auto'):
    fd = open(filename, 'r')  # @reader decorator ensures this is a file descriptor???
    images = list()
    lineindexlist = []
    lineindex = 0
    line = fd.readline()
    lineindex += 1
    while 'begin' in line:
        lineindexlist.append(lineindex)
        line = fd.readline()
        lineindex += 1
        if 'comment' in line:
            comment = line[7:]
            line = fd.readline()
            lineindex += 1

        cell = np.zeros((3, 3))
        for ii in range(3):
            cell[ii] = [float(jj) for jj in line.split()[1:4]]
            line = fd.readline()
            lineindex += 1

        positions = []
        symbols = []
        charges = []  # not used yet
        nn = []  # not used
        forces = []
        energy = 0.0
        charge = 0.0

        while 'atom' in line:
            sline = line.split()
            positions.append([float(pos) for pos in sline[1:4]])
            symbols.append(sline[4])
            nn.append(float(0.0))
            charges.append(float(0.0))
            forces.append([float(pos) for pos in sline[7:10]])
            line = fd.readline()
            lineindex += 1

        while 'end' not in line:
            if 'energy' in line:
                energy = float(line.split()[-1])
            if 'charge' in line:
                charge = 0#float(line.split()[-1])
            line = fd.readline()
            lineindex += 1

        image = Atoms(symbols=symbols, positions=positions, cell=cell)

        sorted_indices = np.argsort(image.numbers)
        image = image[sorted_indices]
        store_energy_and_forces = False
        if with_energy_and_forces == True:
            store_energy_and_forces = True
        elif with_energy_and_forces == 'auto':
            if energy != 0.0 or np.absolute(forces).sum() > 1e-8:
                store_energy_and_forces = True

        if store_energy_and_forces:
            image.calc = SinglePointCalculator(
                atoms=image,
                energy=energy,
                forces=forces,
                charges=charges)
            # charge  = charge)
        images.append(image)
        # to start the next section
        line = fd.readline()
        lineindex += 1

    if index == ':' or index is None:
        return images, lineindexlist
    else:
        return images[index], lineindexlist

def calculate_distance3(hist1, hist2):
    # 当二者范围不等时取1否则取0
    result = [0 if (x == y and x == 0) or (x != 0 and y != 0) else 1 for x, y in zip(hist1, hist2)]
    # 当二者有不同时则取1，否则取0
    return result

def calculate_distance2(hist1, hist2):
    return np.sqrt((hist1 - hist2) ** 2)


def calculate_distance1(hist1, hist2):
    # 当二者范围不等时取1否则取0
    result = [0 if (x == y and x == 0) or (x != 0 and y != 0) else 1 for x, y in zip(hist1, hist2)]
    # 当二者有不同时则取1，否则取0
    return 1 if sum(result) > 1 else 0


def functionname3(glist1, glist2, elementlist, distancemode=0):
    distancelist = {}
    if distancemode == 0:
        for ele in elementlist:
            distancelist[ele] = []

            for i in range(len(glist1[ele])):
                hist1 = glist1[ele][i][0]
                hist2 = glist2[ele][i][0]
                distancelist[ele].append(calculate_distance1(hist1, hist2))
    elif distancemode == 1:
        for ele in elementlist:
            distancelist[ele] = []

            for i in range(len(glist1[ele])):
                hist1 = glist1[ele][i][0]
                hist2 = glist2[ele][i][0]
                distancelist[ele].extend(calculate_distance2(hist1, hist2))
    elif distancemode == 2:
        for ele in elementlist:
            distancelist[ele] = []

            for i in range(len(glist1[ele])):
                hist1 = glist1[ele][i][0]
                hist2 = glist2[ele][i][0]
                distancelist[ele].extend(calculate_distance3(hist1, hist2))

    return distancelist


def getminmaxlist(functionpath):
    max_values = {}
    min_values = {}

    index1 = 0

    with open(functionpath, 'r') as file:
        elementlist = []
        while True:
            line = file.readline().strip()
            if not line or line == '\x1a':
                return max_values, min_values, elementlist
            if line.startswith('#'):  # 忽略以 '#' 开头的行
                continue
            num_atoms = int(line)

            glist2 = {}
            keys = []
            for _ in range(num_atoms):
                atom_info = file.readline().strip().split()
                g_value = list(map(float, atom_info[1:]))
                key = int(atom_info[0])
                keys.append(key)
                if key not in glist2:
                    glist2[key] = []
                glist2[key].append(g_value)
            if len(list(set(keys))) > len(elementlist):
                elementlist = list(set(keys))
                flag = True
            for ele in elementlist:
                if ele in glist2:
                    glist2[ele] = np.array(glist2[ele]).T
                    # 求每一行的最大值
                    max_values1 = np.amax(glist2[ele], axis=1)
                    # 求每一行的最小值
                    min_values1 = np.amin(glist2[ele], axis=1)

                    if flag or ele not in max_values:
                        max_values[ele] = max_values1
                        min_values[ele] = min_values1
                        flag = False
                    else:
                        max_values[ele] = np.maximum(max_values[ele], max_values1)
                        min_values[ele] = np.minimum(min_values[ele], min_values1)

            if functionpath.endswith('.data'):
                file.readline()

            index1 += 1


def getdistancelist(refglist, functionpath, maxvalue, minvalue, intevellist, elementlist, total, intervelnum,
                    distancemode):
    jiangweilist = []
    index1 = 0
    progress_bar = tqdm(range(total))
    with open(functionpath, 'r') as file:
        while True:
            line = file.readline().strip()
            if not line or line == '\x1a':
                progress_bar.close()
                return jiangweilist
            if line.startswith('#'):  # 忽略以 '#' 开头的行
                continue
            num_atoms = int(line)
            molecule = Molecule(num_atoms)
            molecule.index1 = index1

            for _ in range(num_atoms):
                atom_info = file.readline().strip().split()
                atom_number = int(atom_info[0])
                g_value = list(map(float, atom_info[1:]))
                molecule.add_atom(atom_number, g_value)
            if functionpath.endswith('.data'):
                file.readline()
            # distancelist.append(functionname3(refglist, functionname1(molecule, intervel=intervel),
            #                                   intervel))
            m = functionname3(refglist,
                              functionname1(molecule, maxvalue, minvalue, intevellist, elementlist, intervelnum),
                              elementlist, distancemode=distancemode)

            jiangweilist.append([])
            for k in elementlist:
                jiangweilist[index1] = np.concatenate((jiangweilist[index1], m[k]))

            index1 += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"Progress": index1 / total})

def split_list(data_list, split_list):
    sorted_split_list = sorted(split_list)
    result = []
    data_list.pop(0)
    list21 = data_list.copy()
    list22 = data_list.copy()
    for num in sorted_split_list:
        list1=[]
        for item in list21:
            if item <=num:
                list1.append(item)
                list22.remove(item)
        result.append(list1)
        list21=list22
    if len(list21)>0:
        result.append(list21)
    return result


def mplot(data_2d, index_dict, lineindexlist, atomslist, savetype, num):
    fig, ax = plt.subplots()
    x, y = data_2d[:, 0], data_2d[:, 1]

    # plt.scatter(x, y)
    folder_path = os.path.dirname(functionpath)
    libsymbols = ['o', 's', '^', 'v', '*', 'x', '+', '.']
    for i, (system, indices) in enumerate(index_dict.items()):
        indices_lt_num = split_list(indices, num)
        for k, indices_lt in enumerate(indices_lt_num):
            red = (i * 10) % 256 / 256  # Adjusting red component
            green = (i * 20) % 256 / 256  # Adjusting green component
            blue = (i * 30) % 256 / 256  # Adjusting blue component
            alpha = 0.8
            plt.scatter(x[indices_lt], y[indices_lt], s=10, color=(red, green, blue, alpha), marker=libsymbols[k],
                        label=system)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'{mode} Visualization of Multiple Systems')
    plt.xlabel('x-axis \u2212 label')  # 使用Unicode转义序列表示减号符号
    plt.ylabel('y-axis \u2212 label')  # 使用Unicode转义序列表示减号符号

    plt.legend()

    # 调整子图的边距，以便按钮可以自适应窗口大小
    fig.subplots_adjust(bottom=0.2)
    num_buttons = 3
    button_padding = 0.02
    button_width = (1 - (num_buttons + 1) * button_padding) / num_buttons
    button_start = (1 - num_buttons * button_width - (num_buttons - 1) * button_padding) / 2
    buttons = []
    buttonsname = ['Display&Save Structure', 'Delete Structure', 'Output "Input"']
    # 创建按钮对象并排列在一行中
    button_states = [False] * num_buttons
    global dellist
    dellist = np.array([], dtype=int)

    # 添加鼠标点击事件
    def onclick(event):
        global dellist
        if button_states[0] == False and button_states[1] == False:
            return
        elif button_states[0]:
            if event.inaxes is not None:
                # 计算点击位置和所有点之间的距离
                distances = np.sqrt((np.array(x) - event.xdata) ** 2 + (np.array(y) - event.ydata) ** 2)
                # 找到最近点的索引
                closest_index = np.argmin(distances)
                # 检查最近的点是否足够近以视为点击（例如：小于某个阈值）
                if distances[closest_index] < 0.05:
                    # print(f'Index of the closest point: {closest_index}')
                    print(
                        f'对应点位于{os.path.basename(inputpath)}中: {lineindexlist[closest_index]}行,第{closest_index}结构')
                    view(atomslist[closest_index])
                    if savetype is not None:

                        # 创建确认对话框
                        root = tk.Tk()
                        root.withdraw()  # 隐藏根窗口
                        pointx = str(round(x[closest_index], 2))
                        pointy = str(round(y[closest_index], 2))
                        # print(pointx)
                        # print(pointy)
                        # 弹出确认对话框
                        confirmed = messagebox.askyesno("确认保存",
                                                        f"确认将({pointx} , {pointy} )结构{closest_index}保存吗？")
                        if confirmed:
                            try:
                                write(f'{os.path.join(folder_path, str(closest_index))}', atomslist[closest_index],
                                      format=f'{savetype}')
                                print(f'保存{closest_index}成功')
                            except Exception as e:
                                print(f'保存失败：{e}')
                        else:
                            print(f'取消保存{closest_index}')
        else:
            if event.inaxes is not None:
                # 计算点击位置和所有点之间的距离
                distances = np.sqrt((np.array(x) - event.xdata) ** 2 + (np.array(y) - event.ydata) ** 2)
                # 找到范围内的点的索引
                indices_within_range = np.where(distances < 0.05)[0]
                # print(indices_within_range)
                if len(indices_within_range) > 0:

                    # 创建确认对话框
                    root = tk.Tk()
                    root.withdraw()  # 隐藏根窗口
                    pointx = str(round(event.xdata, 2))
                    pointy = str(round(event.ydata, 2))
                    # 弹出确认对话框
                    confirmed = messagebox.askyesno("确认保存",
                                                    f"确认将({pointx} , {pointy} )附近{len(indices_within_range)}个结构加入待删除列表吗？")
                    if confirmed:
                        dellist = np.concatenate((dellist, indices_within_range))
                        ax.scatter(x[indices_within_range], y[indices_within_range], marker='o', facecolors='none',
                                   edgecolors='black', s=10)
                        fig.canvas.draw()
                        # print(dellist)

    def on_button_clicked(event, index):

        if index != num_buttons - 1:
            if index == 0:  # 如果点击的是第一个按钮
                if button_states[1]:  # 如果第二个按钮已经为True，则不进行状态切换
                    button_states[1] = not button_states[1]  # 切换按钮状态
                    buttons[1].color = 'lightblue' if button_states[1] else '0.85'  # 设置按钮颜色
            elif index == 1:  # 如果点击的是第二个按钮
                if button_states[0]:  # 如果第一个按钮已经为True，则不进行状态切换
                    button_states[0] = not button_states[0]  # 切换按钮状态
                    buttons[0].color = 'lightblue' if button_states[0] else '0.85'  # 设置按钮颜色
            button_states[index] = not button_states[index]  # 切换按钮状态
            buttons[index].color = 'lightblue' if button_states[index] else '0.85'  # 设置按钮颜色
            fig.canvas.draw()  # 重新绘制图形

            return button_states[index]  # 返回按钮状态
        else:
            global dellist
            # print(dellist)
            unique_sorted_arr = np.unique(dellist)
            root = tk.Tk()
            root.withdraw()  # 隐藏根窗口
            confirmed = messagebox.askyesno('提示', f"确认删除{len(unique_sorted_arr)}个结构吗？")
            if confirmed:
                delinput(inputpath, dellist)
                pass

    for i in range(num_buttons):
        # 计算按钮的左下角位置
        button_left = button_start + i * (button_width + button_padding)
        button_bottom = 0.01

        # 创建按钮对象
        button_ax = plt.axes([button_left, button_bottom, button_width, 0.075])
        button = Button(button_ax, f'{buttonsname[i]}')

        buttons.append(button)  # 将按钮对象添加到列表中

    # 统一显示按钮
    for i, button in enumerate(buttons):
        button.ax.set_visible(True)
        button.on_clicked(partial(on_button_clicked, index=i))
    # 绑定鼠标点击事件
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event))

    plt.show()


def plot_dist(functionpath, reffunctionpath=None, intervelnum=10, mode='tsne', inputpath=None, savename=None, refnum=1,
              num=-1,
              savetype=None, distancemode=0):
    if reffunctionpath is None:
        reffunctionpath = functionpath
    if refnum < 0:
        refmolecule, _ = read_molecule(reffunctionpath, index=refnum)
        _, total = read_molecule(functionpath, index=refnum)
        if isinstance(num, int) or isinstance(num, float):
            if num < 0:
                num = [total + num + 1]
        elif isinstance(num, list):
            pass
    else:
        refmolecule = read_molecule(reffunctionpath, index=refnum)
        total = read_molecule(functionpath, index=0)
        if isinstance(num, int) or isinstance(num, float):
            if num < 0:
                num = [total + num + 1]
        elif isinstance(num, list):
            pass

    print('***' * 5)
    print(f"当前数据集数量{total}")
    maxvalue, minvalue, elementlist = getminmaxlist(functionpath)
    intevellist = {}
    for ele in elementlist:
        intevellist[ele] = (maxvalue[ele] - minvalue[ele]) / intervelnum
    refglist = functionname1(refmolecule, maxvalue, minvalue, intevellist, elementlist, intervelnum)

    jiangweilist = getdistancelist(refglist, functionpath, maxvalue, minvalue, intevellist, elementlist, total,
                                   intervelnum, distancemode)

    print(f'{mode}降维')

    # 二维坐标
    # 提取 x 和 y 坐标
    if mode.lower() == 'tsne':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=50, learning_rate=10, n_iter=1000)
        data_2d = tsne.fit_transform(np.array(jiangweilist))
    elif mode.lower() == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(jiangweilist)
    elif mode.lower() == 'umap':
        import umap
        # 创建UMAP对象
        umap_obj = umap.UMAP(n_neighbors=2, min_dist=0.3)

        # 对数据进行降维
        data_2d = umap_obj.fit_transform(jiangweilist)
    else:
        sys.exit(-2)

    print('降维结束')
    # 创建一个空的图形对象
    gfig = go.Figure()
    x, y = data_2d[:, 0], data_2d[:, 1]
    gfig.update_layout(
        plot_bgcolor='white',  # 设置绘图区域的背景颜色为白色
        paper_bgcolor='white'  # 设置整个图表的背景颜色为白色
    )

    if inputpath:
        file_extension = os.path.splitext(inputpath)[1]
        if file_extension == '.data':
            # 使用自定义函数read_n2p2读取data文件
            atomslist, lineindexlist = read_n2p2(filename=inputpath, index=':', with_energy_and_forces='auto')
        else:
            # 使用ASE库读取xyz文件
            from ase.io import read
            atomslist = read(inputpath, index=':')
        # atomslist = []

        index_dict = defaultdict(list)
        for i, atoms in enumerate(atomslist):
            index_dict[str(atoms.symbols)].append((i))

        def generate_color_gradient(start_color, end_color, num_steps):
            # Extract RGB components
            r1, g1, b1 = start_color
            r2, g2, b2 = end_color

            colors = []
            for i in range(num_steps):
                # Calculate interpolated RGB values
                new_r = int(r1 + (r2 - r1) * (i / num_steps))
                new_g = int(g1 + (g2 - g1) * (i / num_steps))
                new_b = int(b1 + (b2 - b1) * (i / num_steps))

                # Ensure RGB values are within 0-255 range
                new_r = min(max(new_r, 0), 255)
                new_g = min(max(new_g, 0), 255)
                new_b = min(max(new_b, 0), 255)

                # Convert to Plotly color format (#RRGGBB)
                color = f'#{new_r:02X}{new_g:02X}{new_b:02X}'

                colors.append(color)

            return colors

        start_color = (237, 237, 214)
        end_color = (104, 166, 124)
        num_steps=len(index_dict)
        colors = generate_color_gradient(start_color, end_color, num_steps)


        pltlysymbols = ['circle', 'square', 'triangle-up', 'triangle-down', 'star', 'x', 'cross', 'dot']
        pltlysymbolsize = [8, 8, 8, 8, 8, 8, 8, 8]

        for i, (system, indices) in enumerate(index_dict.items()):
            color = colors[i]

            indices_lt_num = split_list(indices, num)
            for k, indices_lt in enumerate(indices_lt_num):
                gfig.add_trace(go.Scatter(
                    x=x[indices_lt],
                    y=y[indices_lt],
                    mode='markers',
                    name=system,  # 为每个散点图设置标签，它会显示在图例中
                    marker=dict(color=color, symbol=pltlysymbols[k], size=pltlysymbolsize[k])
                ))

    else:
        gfig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
        ))

    # 设置图表的标题和坐标轴标签
    gfig.update_layout(
        title=f'{mode} Visualization of Multiple Systems',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        legend_title='Categories'
    )

    # 绘制图表
    if savename:
        plot(gfig, filename=f'{savename}.html')
    else:
        import datetime
        now = datetime.datetime.now()
        folder_path = os.path.dirname(functionpath)
        filename = os.path.basename(functionpath).split('.')[0]
        file_name = f'{now.strftime("%Y%m%d%H%M")}_{filename}_{intervelnum}_{mode}_{distancemode}.html'
        #plot(gfig, filename=os.path.join(folder_path, file_name))
        import plotly.io as pio

        # 假设 gfig 是 Plotly 图形对象，folder_path 是保存路径，file_name 是文件名
        file_path = os.path.join(folder_path, file_name)

        # 将图形保存为 HTML 文件
        pio.write_html(gfig, file_path)
        np.save(
            os.path.join(folder_path, f'jiangweilist_b_{now.strftime("%Y%m%d%H%M")}_{filename}_{intervelnum}_{mode}_{distancemode}'),
            jiangweilist)
        np.save(os.path.join(folder_path, f'data_2d_b_{now.strftime("%Y%m%d%H%M")}_{filename}_{intervelnum}_{mode}_{distancemode}'),
                data_2d)
        csv_path='data_2d_b.csv'
        np.savetxt(csv_path, data_2d, delimiter=',', fmt='%f')
        np.save(os.path.join(folder_path,
                             f'index_dict_{now.strftime("%Y%m%d%H%M")}_{filename}_{intervelnum}_{mode}_{distancemode}'),
                index_dict)

        print(f'保存{os.path.join(folder_path, file_name)}')
    print('显示')
    # if inputpath:
    #     mplot(data_2d, index_dict, lineindexlist, atomslist, savetype, num)


def delinput(path, dellist):
    with open(path, "r") as f:
        lines = f.readlines()

    structures = []
    in_structure = False
    current_structure = ""

    for line in lines:
        if line.startswith("begin"):
            in_structure = True
            current_structure = ""
        elif line.startswith("end"):
            in_structure = False
            current_structure += line
            structures.append(current_structure)
        if in_structure:
            current_structure += line

    new_structures = np.delete(structures, dellist)
    directory = os.path.dirname(path)
    new_path = os.path.join(directory, "newinput.data")
    with open(new_path, "w") as f:
        for structure in new_structures:
            f.write(structure)

    print(new_path, '保存成功')


def find_latest_file(pattern):
    # 使用glob模块获取匹配的所有文件
    files = glob.glob(pattern)
    if not files:
        return None
    # 对文件按照修改时间排序，选择最新的文件
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def mainf(functionpath, reffunctionpath, intervelnum=10, mode='tsne', inputpath=None, savename=None, refnum=1, num=-1,
          savetype=None, distancemode=0):
    folder_path = os.path.dirname(functionpath)
    filename = os.path.basename(functionpath).split('.')[0]

    # 构建文件名的模式
    pattern1 = f'{folder_path}\\*_{filename}_{intervelnum}_{mode}_{distancemode}.html'
    latest_file = find_latest_file(pattern1)

    plot_dist(functionpath=functionpath, reffunctionpath=reffunctionpath, intervelnum=intervelnum, mode=mode,
                  inputpath=inputpath, savename=savename,
                  refnum=refnum, num=num, savetype=savetype, distancemode=distancemode)


if __name__ == '__main__':
    a = time.time()
    functionpath = r'function.data'
    reffunctionpath = r'function.data'
    # reffunctionpath = None
    inputpath = r'input.data'  # input文件地址
    # inputpath = None
    savename = None  # None自动保存在 functionpath
    num = -1  # 用于区分新结构，num值为旧结构的数量，无新结构用-1(要求新结构拼接在旧数据后面)
    refnum = 1  # 对比参考结构，默认为1，即与第一个对比。取负数-x为最后第x个取。
    # intervelnum = 20  # 区间数量
    mode = 'tsne'  # pca或tsne
    distancemodelist = [2]  # 差异向量类别，类别0无差异则用0否则用1，类别1将向量距离作为差异向量值，默认类别0

    intervelnumlist=[25]
    for intervelnum in intervelnumlist:
        for distancemode in distancemodelist:
            mainf(functionpath=functionpath,reffunctionpath=reffunctionpath, intervelnum=intervelnum, mode=mode, inputpath=inputpath, savename=savename,
              refnum=refnum, num=num, savetype='vasp', distancemode=distancemode)
    print("Time used:", time.time() - a)

    pass

