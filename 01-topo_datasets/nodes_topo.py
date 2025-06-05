import numpy as np
import pandas as pd
from _ImportHyperGraph import HyperGraph
import os
import time
from tqdm import tqdm
from Hyperspreading import Hyperspreading
from Fitness import fitness
def cal_Adeff(dianqiangdu_list,dianchaodu_list,num_node):
    return np.array([dianqiangdu_list[i]/dianchaodu_list[i] if dianchaodu_list[i] else 0 for i in range(num_node) ])
def cal_MC(df_hyper_matrix, seeds, t, b):
    I_list = list(seeds)
    beta = b
    iters = t
    I_total_list = [len(I_list)]
    for t in range(0, iters):
        infected_T = []
        for inode in I_list:
            adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
            infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
            infected_T.extend(infected_list_unique)
        I_list.extend(infected_T)
        I_total_list.append(len(I_list))
    # plt.plot(np.arange(np.array(len(I_total_list))),I_total_list,color='orange')
    # plt.show()
    return I_total_list[-1]

def cal_HEVD(df_hyper_matrix, seed_set, beta=0.01):
    # 只考虑一跳范围内传播节点数量的期望
    # 1、首先确定初始种子集的一跳区域（除开种子外的）
    one_hop_nodes = fitness.get_one_hop_Nodes_of_Nodes(seed_set, df_hyper_matrix)  # 获取所有一跳区域节点
    one_hop_nodes_set = [x for x in one_hop_nodes if x not in seed_set]  # 除去种子节点部分

    # 2、遍历每一个邻居节点i，计算其被感染的概率，求和
    EDV = len(seed_set)
    for i, inode in enumerate(one_hop_nodes_set):
        adj_inodes_list = fitness.get_one_hop_Nodes_of_Node(inode, df_hyper_matrix)  # 该邻居节点的邻居
        seeds_involved = [x for x in adj_inodes_list if x in seed_set]  # 与该节点相连的种子节点
        pro_not_i = 1
        for j in seeds_involved:
            # 两个点之间的传染概率并不是beta，而是什么呢？？？beta*某个种子节点与该节点共有的超边数/这个种子节点自身所有的超边数
            pro_not_i *= (1 - beta * fitness.cal_select_probability(inode, j, df_hyper_matrix))
        pro_i = 1 - pro_not_i
        EDV += pro_i
    return EDV


def GIwCC(num_nodes, df_adj_matrix, df_c):
    GIwCC1_list = []
    neighbors_list = []
    for i in tqdm(range(num_nodes),desc='1'):
        neibors = Hyperspreading.getNodesofHpe(i, df_adj_matrix)
        GIwCC1_list.append(len(neibors))
        neighbors_list.append(neibors)
    GIwCC2_list = GIwCC1_list.copy()
    for i in tqdm(range(num_nodes),desc='2'):
        for j in neighbors_list[i]:
            GIwCC2_list[i] += GIwCC1_list[j] * df_c
    CIwCC3_list = GIwCC1_list.copy()
    for i in tqdm(range(num_nodes),desc='3'):
        for j in neighbors_list[i]:
            CIwCC3_list[i] += GIwCC2_list[j] * df_c
    GIwCC3_dict = {}
    for i in range(num_nodes):
        GIwCC3_dict[i] = CIwCC3_list[i]
    return GIwCC3_dict


'''当我想要新加一个节点的指标值时
    我需要：
    1.设计计算函数
    2.读取现有表格的指标
    3.合并新旧指标
    4.写入表格'''
def newdata(filepath,outpath):

    H = HyperGraph(filepath)
    H.dataload()

    neibor_matrix = H.neibor_matrix
    relevance_matrix = H.relevance_matrix
    num_node, num_edge = H.num_node, H.num_edge
    # dianqiangdu_list = H.dianqiangdu_list
    # dianchaodu_list = H.dianchaodu_list
    # diandu_list = H.diandu_list

    # Adeff_dict = cal_Adeff(dianqiangdu_list,dianchaodu_list,num_node)
    MC_dict,HEDV_dict={},{}
    node_list=list(range(num_node))
    for i in node_list:
        mci=0
        for j in range(500):
            mci+=cal_MC(relevance_matrix, [i], 1, 0.01)
        MC_dict[i] = mci/500
        HEDV_dict[i] = cal_HEVD(relevance_matrix, [i], beta=0.01)

    # df = pd.read_excel('../01-topo_datasets/nodes_topo_result/%s.xlsx' % dataset[:-4])
    df = pd.DataFrame()
    df['MC=1'] = MC_dict
    df['HEDV'] = HEDV_dict

    # df['deg'] = diandu_list
    # df['d^H'] = dianchaodu_list
    # df['d^S'] = dianqiangdu_list

    df.to_excel(outpath)
def olddata(filepath,outpath):

    H = HyperGraph(filepath)
    H.dataload()

    neibor_matrix = H.neibor_matrix
    relevance_matrix = H.relevance_matrix
    num_node, num_edge = H.num_node, H.num_edge
    dianqiangdu_list = H.dianqiangdu_list
    dianchaodu_list = H.dianchaodu_list
    diandu_list = H.diandu_list

    #
    # Adeff_dict = cal_Adeff(dianqiangdu_list,dianchaodu_list,num_node)
    df = pd.read_excel(outpath)
    # MC_dict = {}
    # node_list = list(range(num_node))
    # I = len(node_list)
    # for i in tqdm(node_list):
    #     mci = 0
    #     for j in range(500):
    #         mci += cal_MC(relevance_matrix, [i], 30, 0.01)
    #         # print(j)
    #     MC_dict[i] = mci / 2000
        # print(i)

    # df['MC'] = MC_dict
    # df['Adeff_dict'] = Adeff_dict
    # df['deg'] = diandu_list
    # df['d^H'] = dianchaodu_list
    # df['d^S'] = dianqiangdu_list

    df['GIwCC0.2'] = GIwCC(num_node, neibor_matrix, 0.2)
    df.to_excel(outpath)

filepath = '../0-hypergraph_datasets/Algebra.txt'
outpath = '../01-topo_datasets/nodes_topo_result/Algebra.xlsx'
olddata(filepath,outpath)



import matplotlib.pyplot as plt
def HEDVvsMC():
    H = HyperGraph(filepath)
    H.dataload()
    neibor_matrix = H.neibor_matrix
    relevance_matrix = H.relevance_matrix
    num_node, num_edge = H.num_node, H.num_edge

    MC_value, HEDV_value = [], []
    MC_time, HEDV_time = [], []
    node_list = list(range(num_node))
    # 计时开始
    TMC = time.time()
    mc = 0
    for i in range(10000):
        mc += cal_MC(relevance_matrix, [4, 9], 1, 0.01)
        MC_value.append(mc / (i + 1))
        # 每次迭代添加时间到MC_time
        MC_time.append(time.time() - TMC)

    THEDV = time.time()
    HEDV_value = [(cal_HEVD(relevance_matrix, [4, 9], beta = 0.01))]
    HEDV_time = [(time.time() - THEDV)]
    HEDV_value = HEDV_value * 10000
    HEDV_time = HEDV_time * 10000
    # ================== 学术图表规范设置 ==================
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # 使用期刊推荐的字体
        'font.size': 12,         # 正文字号通常10-12pt
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 600,       # 满足期刊分辨率要求
        'savefig.dpi': 600,
        'lines.linewidth': 1.8,  # 适当增加线宽
        'axes.linewidth': 1.2,   # 坐标轴线宽
        'grid.linestyle': '--',
        'grid.alpha': 0.4
    })

    # ================== 创建画布和坐标轴 ==================
    fig, ax1 = plt.subplots(figsize=(8, 5))  # 适合双栏排版的尺寸

    # ================== 主坐标轴设置（左侧） ==================
    ax1.set_xlabel('Simulation Counts', labelpad=10)
    ax1.set_ylabel('Influence Value', color='#2a2a2a', labelpad=10)
    l1, = ax1.plot(MC_value, color='#264653', alpha=0.9, label='Monte Carlo')
    l2, = ax1.plot(HEDV_value, color='#55b0a2', alpha=0.9, ls='-', label='HEDV')
    ax1.tick_params(axis='y', which='both', direction='in', width=1.2)

    # ================== 次坐标轴设置（右侧） ==================
    ax2 = ax1.twinx()
    ax2.set_ylabel('Simulation Time (s)', color='#2a2a2a', labelpad=10)
    l3, = ax2.plot(MC_time, color='#e8c369', alpha=0.8, label='MC Time')
    l4, = ax2.plot(HEDV_time, color='#e67357', alpha=0.8, label='HEDV Time')
    ax2.tick_params(axis='y', which='both', direction='in', width=1.2)

    # ================== 图例专业排版 ==================
    # 左上角图例（Value）
    leg1 = ax1.legend(handles=[l1, l2],
                    loc='upper left',
                    bbox_to_anchor=(0.05, 0.98),
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='#2a2a2a',
                    title='Influence Metrics:')

    # 右下角图例（Time）
    leg2 = ax2.legend(handles=[l3, l4],
                    loc='lower right',
                    bbox_to_anchor=(0.98, 0.05),
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='#2a2a2a',
                    title='Time Cost:')

    # ================== 网格和样式优化 ==================
    ax1.grid(True, which='major', axis='both', linestyle='--', alpha=0.4)
    ax2.grid(False)  # 避免右侧网格干扰

    # ================== 坐标轴刻度优化 ==================
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.2)
        ax2.spines[axis].set_linewidth(1.2)

    # ================== 标题和保存设置 ==================
    # plt.title('Comparative Analysis of Influence Spread Algorithms\nMC vs HEDV in Hypergraph Networks',
    #          pad=20,
    #          fontweight='semibold')

    # 保存矢量格式（投稿推荐）
    plt.tight_layout(pad=2.0)
    plt.savefig('Algorithm_Comparison.png', dpi= 600, bbox_inches='tight')
    plt.savefig('Algorithm_Comparison.pdf', format='pdf', bbox_inches='tight')
    plt.show()



'''filepath = 'D:/毕业设计/01-Experiments/0-hypergraph_datasets/'
# file_name = os.listdir(filepath)
file_name = ['MyExample.txt']
outpath = 'D:/毕业设计/01-Experiments/01-topo_datasets/nodes_topo_result/'
for i in file_name:
    # print(file_name)
    newdata(filepath+i,outpath+i[:-4]+'.xlsx')
    # olddata(filepath+i,outpath+i[:-4]+'.xlsx')'''