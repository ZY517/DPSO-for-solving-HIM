from Dataloader import dataloader
from t_Al import algorithms
import random
import os
import numpy as np
import pandas as pd
import math


if __name__ == '__main__':
    # 1、获取路径下面所有数据集的名称

    datasets_name = os.listdir('../0-hypergraph_datasets/Ori_data/')
    # n = 0
    # 2、依次读取这些文件，并对每一个数据使用不同算法进行搜索'email-Enron.txt'
    # for dataset in ['Algebra.txt']:
    for dataset in datasets_name[:]:
        # if dataset == 'diseasome.txt':
        #     continue
        randomseed = 55
        random.seed(randomseed)
        path = '../0-hypergraph_datasets/Ori_data/' + dataset
        print('-------------------------Searching %d---%s-------------------------' %(randomseed, dataset))
        # 2.1 调用dataloader类的dataload方法，获取必要信息
        dl = dataloader(path)
        dl.dataload()
        df_hyper_matrix = dl.hyper_matrix  # 从数据中获取的节点超边矩阵
        df_adj_matrix = dl.adj_matrix

    # 3、设置K的值
        K = 30  # 目标种子集合的大小

        # 4、使用每一种已有的算法获取种子节点
        # 4.1 创建容器
        seeds_list = []
        cost_time_list = []
        percost_time_list = []
        pergbest_list = []
        # 4.2 使用每一种算法求解
        '''
        顺序依次为
        DPSO, HADP, HSDP, H-RIS, H-CI(I=1), H-CI(I=2), H-Degree, Degree, General-greedy
        '''

        methods = ['DPSO', 'HADP', 'HSDP', 'H-RIS', 'H-CI(I=1)', 'H-CI(I=2)', 'H-Degree','Hyper-IMRank']




        seeds_list_DPSO, cost_time_DPSO, gbest_list, percost_time_list_DPSO = algorithms.DPSO(df_hyper_matrix.values, K, 'HEDV', 0.7, randomseed = randomseed)#527注释
        seeds_list_HADP, cost_time_HADP = algorithms.HADP(df_hyper_matrix, K)
        seeds_list_HSDP, cost_time_HSDP = algorithms.HSDP(df_hyper_matrix, K)
        seeds_list_RIS, cost_time_RIS = algorithms.RIS(df_hyper_matrix, K, 0.01, 200)
        seeds_list_CI1, cost_time_CI1 = algorithms.CI(df_hyper_matrix, K, 1)
        seeds_list_CI2, cost_time_CI2 = algorithms.CI(df_hyper_matrix, K, 2)
        seeds_list_HDegree, cost_time_HDegree = algorithms.HDegree(df_hyper_matrix, K)
        seeds_list_HyIMR, cost_time_HyIMR = algorithms.HyperIMRANK(df_hyper_matrix, K)
        seeds_list_Degree, cost_time_Degree = algorithms.degreemax(df_hyper_matrix, K)
        seeds_list_Greedy, cost_time_Greedy = algorithms.generalGreedy(df_hyper_matrix, K, mtkl=50)
        # 4.3 全部添加到列表中，统一输出


        seeds_list_DPSO.append(seeds_list_DPSO)
        seeds_list.append(seeds_list_HyIMR)
        seeds_list.append(seeds_list_HADP)
        seeds_list.append(seeds_list_HSDP)
        seeds_list.append(seeds_list_RIS)
        seeds_list.append(seeds_list_CI1)
        seeds_list.append(seeds_list_CI2)
        seeds_list.append(seeds_list_HDegree)
        seeds_list.append(seeds_list_Greedy)

        cost_time_list.append(cost_time_DPSO)
        cost_time_list.append(cost_time_HyIMR)
        cost_time_list.append(cost_time_HADP)
        cost_time_list.append(cost_time_HSDP)
        cost_time_list.append(cost_time_RIS)
        cost_time_list.append(cost_time_CI1)
        cost_time_list.append(cost_time_CI2)
        cost_time_list.append(cost_time_HDegree)
        cost_time_list.append(cost_time_Greedy)

        # 5、保存搜索到的种子结果
        seeds_result = pd.DataFrame(seeds_list).T
        seeds_result = pd.DataFrame(seeds_list).T
        seeds_result.loc[len(seeds_result.index)] = np.array(cost_time_list)
        seeds_result.index = list(np.arange(1, K + 1)) + ['Time_Cost']
        seeds_result.columns = methods
        seeds_result.to_excel('./seeds_result/' + dataset[:-4] + '.xlsx', sheet_name = "Seeds_List")
        print('-------------------------Finished-------------------------\n')


        ######################################### 保留DPSO的群体最优和时间开销 ##############################################
        # seeds_list.append(seeds_list_DPSO)
        # percost_time_list.append(percost_time_list_DPSO)
        # pergbest_list.append(gbest_list)
        # cost_time_list.append(cost_time_DPSO)
        # seeds_result = pd.DataFrame(seeds_list).T
        # percost_time_result = pd.DataFrame(percost_time_list).T
        # pergbest_list_result = pd.DataFrame(pergbest_list).T
        #
        # seeds_result.loc[len(seeds_result.index)] = np.array(cost_time_list)
        # seeds_result.index = list(np.arange(1, K + 1)) + ['Time_Cost']
        # seeds_result.columns = methods
        #
        # percost_time_result.loc[len(seeds_result.index)] = np.array(cost_time_list)
        # percost_time_result.index = list(np.arange(1, K + 1)) + ['Time_Cost']
        # percost_time_result.columns = methods
        #
        # pergbest_list_result.index = list(np.arange(1, K + 1))
        #
        # with pd.ExcelWriter('./DPSO_Adaptive_c/' + dataset[:-4] + '.xlsx',
        #                     mode = 'a',
        #                     engine = 'openpyxl',
        #                     if_sheet_exists = 'replace') as writer:
        #     seeds_result.to_excel(writer, sheet_name = "Seeds_List", index = True)
        #     percost_time_result.to_excel(writer, sheet_name = "Per_Cost_Time", index = True)
        #     pergbest_list_result.to_excel(writer, sheet_name = "Per_Gbest_List", index = True)
        #
        # print('-------------------------Finished-------------------------\n')




