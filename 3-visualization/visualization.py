import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from dataload import dataload
from tqdm import tqdm
# 1.3 字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'cm' # 公式的字体为

colors_plot = [
'#E25C5C',
'#82B0D2',
'#F3D266',
'#BECFC9',
'#C1DE9C',
'#E7DAD2',
'#BEB8DC',
'#999999',
'#FFB061',
'#f4d8dd',
]

# marker_symbols = ['s-', 'd-', '^-', 'v-', '*-', 'H-', '+-', 'x-', '--','2-' ]
marker_symbols = ['1-', '2-', '3-', '4-', '^-', 'v-', '+-', 'x-','*-']

class visualization:
    def k_scale_curve(x):
        Datasets = os.listdir('../0-hypergraph_datasets/Ori_data/')
        Algorithms = ['DPSO-HEDV_NG40H1', 'DPSO_NG40H1', 'HEDV-greedy', 'HADP', 'HSDP', 'H-RIS', 'H-CI(I=2)','H-Degree', 'Hyper-IMRank' ]
        labels = ['DPSO-roulette', 'DPSO-random', 'HEDV-greedy', 'HADP', 'HSDP', 'H-RIS',  'H-CI','H-Degree', 'Hyper-IMRank']

        # 创建画布和子图布局
        fig = plt.figure(figsize = (18, 10))
        gs = fig.add_gridspec(2, 4, hspace = 0.3, wspace = 0.25,
                              top = 0.85, bottom = 0.05,
                              left = 0.07, right = 0.93)

        beta = 0.01
        t = 25
        # 用于存储所有线条的句柄和标签

        for i, filename in tqdm(enumerate(Datasets), desc = 'Datasets'):
            data = dataload.get_scale(filename[:-4], beta)
            x_index_for_k = data.index

            ax = fig.add_subplot(gs[i // 4, i % 4])
            ax.set_xlabel('$K$', fontsize = 16)
            ax.set_ylabel('Propagation scale', fontsize = 16)
            ax.set_title(filename[:-4], fontsize = 16, pad = 10)
            ax.grid(True, alpha = 0.3, linestyle = ':')
            for j, name in enumerate(Algorithms):
                y_index_for_scale = [scale_list.mean(axis = 0)[t] for scale_list in data[name]]
                line, = ax.plot(x_index_for_k, y_index_for_scale, marker_symbols[j], \
                                color = colors_plot[j], label = labels[j], zorder = len(labels) - j, mfc = 'None',
                                mew = 1.5)
            # break
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, ncol=10, bbox_to_anchor=(0.5, 0.955), loc='upper center', \
                   prop={'size': 16}, frameon=False)
        # plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/DPSO-HEDV_K.jpg', dpi = 600, bbox_inches = 'tight')
        # plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/DPSO-HEDV_K.svg', dpi = 600, bbox_inches = 'tight')
        plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/NEW1a.svg', dpi = 600, bbox_inches = 'tight')
        # print('ok')

    def t_scale_curve(self):
        # 2、数据读取
        Datasets = os.listdir('../0-hypergraph_datasets/Ori_data/')
        # Datasets = ['email-Enron.txt']
        Algorithms = ['DPSO-HEDV_NG40H1','DPSO_NG40H1', 'HEDV-greedy', 'HADP', 'HSDP', 'H-RIS', 'H-CI(I=2)', 'H-Degree','Hyper-IMRank']

        labels = ['DPSO-roulette','DPSO-random','HEDV-greedy', 'HADP', 'HSDP', 'H-RIS', 'H-CI', 'H-Degree', 'Hyper-IMRank']

        # 创建画布和子图布局
        fig = plt.figure(figsize = (18, 10))
        gs = fig.add_gridspec(2, 4, hspace = 0.3, wspace = 0.25,
                              top = 0.85, bottom = 0.05,
                              left = 0.07, right = 0.93)
        
        k = 30
        beta = 0.01
        t_max = 25

        for i, filename in tqdm(enumerate(Datasets)):
            data = dataload.get_scale(filename[:-4], beta, )
            x_index_for_t = range(t_max)

            ax = fig.add_subplot(gs[i // 4, i % 4])
            ax.set_xlabel('$T$', fontsize=16)
            ax.set_ylabel('Propagation scale', fontsize=16)
            ax.set_title(filename[:-4], fontsize = 16, pad = 10)
            ax.grid(True, alpha = 0.3, linestyle = ':')

            for j, name in enumerate(Algorithms):
                y_index_for_scale = data[name].loc[k].mean(axis=0)[:t_max]
                line, = ax.plot(x_index_for_t, y_index_for_scale, marker_symbols[j], \
                                 color=colors_plot[j], label=labels[j], zorder=len(labels) - j, mfc='None', mew=1.5)

        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, ncol=10, bbox_to_anchor=(0.5, 0.95), loc='upper center', \
                   prop={'size': 16}, frameon=False)

        # plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/DPSO-HEDV_T.jpg', dpi=600, bbox_inches='tight')
        # plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/DPSO-HEDV_T.svg', dpi=600, bbox_inches='tight')

        plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/NEW1b.svg', dpi=600, bbox_inches='tight')
        # print('ok')

    def K_time_curve(self):
        # 1、外观设置
        # 1.1 颜色
        color_name = 'Paired'
        colors_plot = list(matplotlib.colormaps.get_cmap(color_name).colors)
        # element = colors_plot.pop(5)  # 移除第6个元素，并将其保存到变量element中
        # colors_plot.insert(0, element)  # 将保存的元素插入到列表的开头
        element = colors_plot[5]  # 移除第6个元素，并将其保存到变量element中
        colors_plot.insert(0, element)  # 将保存的元素插入到列表的开头
        # 1.2 形状
        marker_symbols = ['--', 'p-', 'd-', 's-' ]
        # 1.3 字体
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 11

        # 2、数据读取
        Datasets = os.listdir('../0-hypergraph_datasets/Ori_data/')
        # Datasets = ['email-Enron.txt']
        Algorithms = ['DPSO-HEDV_NG40H1','DPSO-HEDV', 'HEDV-Greedy', 'HADP', ]

        labels = ['DPSO-HEDV_NG40H1','DPSO-HEDV', 'HEDV-greedy', 'HADP' ]
        fig = plt.figure(figsize = (18, 10))
        gs = fig.add_gridspec(2, 4, hspace = 0.3, wspace = 0.25,
                              top = 0.88, bottom = 0.08,
                              left = 0.07, right = 0.93)
        k = 50
        beta = 0.01
        filepath = '../1-search_seeds/cost_time_result/cost_time_result/'
        for i, filename in tqdm(enumerate(Datasets)):
            ax = fig.add_subplot(gs[i // 4, i % 4])
            data = pd.read_excel(filepath+filename[:-4]+'.xlsx')
            # data2 = pd.read_excel(filepath+filename[:-4]+'_50.xlsx')
            x_index_for_t = range(1,k+1)

            ax.set_xlabel('K')
            ax.set_ylabel('Time Cost')

            ax.set_title(filename[:-4])
            # ax.text(0.5,0.5,'(%s)'%(i+1),fontsize=18,ha='center',zorder=100)

            for j, name in enumerate(Algorithms):
                y_index_for_scale = data[name][:-1]

                ax.plot(x_index_for_t, y_index_for_scale, marker_symbols[j], \
                                 color=colors_plot[j], label=labels[j], zorder=len(labels) - j)

            # for sad,name in enumerate(['HEDV-Greedy']):
            #     y_index_for_scale = data2[name][:-1]
            #
            #     ax.plot(x_index_for_t, y_index_for_scale, marker_symbols[j], \
            #                      color=colors_plot[j], label='HEDV-greedy', zorder=len(labels) - j)
                # plt.plot(x_index_for_k, y_index_for_scale)

        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, ncol=4, bbox_to_anchor=(0.51, 0.96), loc='upper center', \
                   prop={'size': 13.3}, frameon=False)
        # plt.savefig('T_scale_curve.svg', dpi = 400, bbox_inches = 'tight')

        plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/4_KTime.png', dpi=800, bbox_inches='tight')
        plt.show()
    def NB_graph(self):
       
        # 2、数据读取
        Datasets = os.listdir('../0-hypergraph_datasets/Ori_data/')
            
        labels = ['HEDV-greedy','DPSO-roulette','DPSO-random','HADP']

        # 创建画布和子图布局
        fig = plt.figure(figsize = (18, 10))
        gs = fig.add_gridspec(2, 4, hspace = 0.3, wspace = 0.25,
                              top = 0.85, bottom = 0.05,
                              left = 0.07, right = 0.93)
        
        k = 30
        beta = 0.010
        t_max = 25
        mtkl = 2000
        alpha_mtkl = 0.15
        
        for i, filename in tqdm(enumerate(Datasets)):
            ax = fig.add_subplot(gs[i // 4, i % 4])
            data = dataload.get_scale(filename[:-4], beta)
            DPSO_HEDV_all = np.array([x[:t_max] for x in data['DPSO-HEDV_NG40H1'].iloc[-1]])[:mtkl]
            DPSO_HEDV_avg = DPSO_HEDV_all.mean(axis = 0)

            DPSO_all = np.array([x[:t_max] for x in data['DPSO_NG40H1'].iloc[-1]])[:mtkl]
            DPSO_avg = DPSO_all.mean(axis=0)

            HEDV_Greedy_all = np.array([x[:t_max] for x in data['HEDV-greedy'].iloc[-1]])[:mtkl]
            HEDV_Greedy_avg = HEDV_Greedy_all.mean(axis=0)

            HADP_all = np.array([x[:t_max] for x in data['HADP'].iloc[-1]])[:mtkl]
            HADP_avg = HADP_all.mean(axis = 0)

            x_index_for_t = np.array(range(t_max))

            ax.set_xlabel('$T$', fontsize = 16)
            ax.set_ylabel('Propagation scale', fontsize = 16)
            ax.set_title(filename[:-4], fontsize = 16, pad = 10)
            ax.grid(True, alpha = 0.3, linestyle = ':')

            # 1、先画两个的所有曲线
            DPSO_HEDV_all_A, DPSO_HEDV_all_B = visualization.concat_array(x_index_for_t, DPSO_HEDV_all)
            DPSO_all_A, DPSO_all_B = visualization.concat_array(x_index_for_t, DPSO_all)
            HEDV_Greedy_all_A , HEDV_Greedy_all_B = visualization.concat_array(x_index_for_t, HEDV_Greedy_all)
            HADP_all_A , HADP_all_B = visualization.concat_array(x_index_for_t, HADP_all)

            
            #红 蓝 紫 泥
            line1, = ax.plot(HEDV_Greedy_all_A, HEDV_Greedy_all_B, color = colors_plot[2], alpha = alpha_mtkl,linewidth = 0.2)
            line2, = ax.plot(DPSO_HEDV_all_A, DPSO_HEDV_all_B, color=colors_plot[0], alpha = alpha_mtkl, linewidth = 0.2)
            line3, = ax.plot(DPSO_all_A, DPSO_all_B, color=colors_plot[1], alpha=alpha_mtkl, linewidth=0.2)

            # ax.plot(HADP_all_A, HADP_all_B, color=(230/255,111/255,81/255), alpha = alpha_mtkl, linewidth = 0.2)


            # 1、再绘制两个的平均曲线
            line1, = ax.plot(x_index_for_t, HEDV_Greedy_avg, color=colors_plot[2], linewidth = 2, label = 'HEDV-greedy')
            line2, = ax.plot(x_index_for_t, DPSO_HEDV_avg, color=colors_plot[0], linewidth = 2, label = 'DPSO-roulette')
            line3, = ax.plot(x_index_for_t, DPSO_avg, color=colors_plot[1], linewidth = 2, label = 'DPSO-random')

            # ax.plot(x_index_for_t, HADP_avg, color=(230/255,111/255,81/255), linewidth=3, label='HADP')

        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines[::-1], labels[::-1], ncol=3, bbox_to_anchor=(0.51, 0.95) ,loc = 'upper center',\
                   prop = {'size':16}, frameon=False, handletextpad=1, columnspacing=3)
        # handlelength=3 设置了句柄的长度为3, handletextpad=1.5 设置了句柄和文本之间的间距为1.5,columnspacing=2 设置了图例列之间的间距为2
        # plt.savefig('NB_graph.svg', dpi = 400, bbox_inches = 'tight').

        plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/15NB3_T_k=25.jpg', dpi=600, bbox_inches='tight')
        plt.savefig('D:/MyOneDrive/OneDrive/桌面/Figure/15NB3_T_k=25.svg', dpi=600, bbox_inches='tight')

        # plt.show()

    def array_2d_to_mean(array):
        return np.array(array).mean(axis = 0)
    
    def concat_array(x_index, y_index_list):
        t_max = len(x_index)
        x_index = x_index.astype("float")
        y_index_list = y_index_list.astype("float")
        # 1、处理横坐标
        # 每相隔x_index添加一个np.nan
        A = np.tile(np.append(x_index,[np.nan]),len(y_index_list))
        # 2、处理纵坐标
        # 把y_index_list拉开成一个一维向量
        raveled_y = y_index_list.flatten()
        # 每相隔t_max添加一个np.nan
        # 计算要插入的元素个数
        num_nans = len(y_index_list)
        # 生成要插入的元素，都为np.nan
        nans = np.full(num_nans, np.nan)
        # 使用np.insert()函数在原数组的每隔2个元素的位置插入np.nan
        B = np.insert(raveled_y, np.arange(t_max, len(raveled_y)+1, t_max), nans)
        return A,B
 
if __name__ == '__main__':
    visualization.k_scale_curve(9)
    visualization.t_scale_curve(9)
    # visualization.NB_graph(3)
    # visualization.K_time_curve(1)