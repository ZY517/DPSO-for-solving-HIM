from Dataloader import dataloader
from t_Al import algorithms
import random
import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # 1、获取路径下面所有数据集的名称

    datasets_name = os.listdir('../0-hypergraph_datasets/Ori_data/')
    # n = 0
    # 2、依次读取这些文件，并对每一个数据使用不同算法进行搜索
    # for dataset in :'email-Enron.txt',['diseasome.txt']
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
        H_list = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
        for H in H_list:
        # 3、设置K的值
            K = 5  # 目标种子集合的大小

            # 4、使用每一种已有的算法获取种子节点
            # 4.1 创建容器
            seeds_list = []
            cost_time_list = []
            percost_time_list = []
            pergbest_list = []
            # 4.2 使用每一种算法求解

            methods = ['DPSO-HEDV']


            seeds_list_DPSO, cost_time_DPSO, gbest_list, percost_time_list_DPSO = algorithms.DPSO(df_hyper_matrix.values, K, 'HEDV', H, randomseed = randomseed)

            # 4.3 全部添加到列表中，统一输出
            seeds_list.append(seeds_list_DPSO)
            percost_time_list.append(percost_time_list_DPSO)
            pergbest_list.append(gbest_list)


            cost_time_list.append(cost_time_DPSO)


            # 5、保存搜索到的种子结果
            seeds_result = pd.DataFrame(seeds_list).T
            percost_time_result = pd.DataFrame(percost_time_list).T
            pergbest_list_result = pd.DataFrame(pergbest_list).T

            seeds_result.loc[len(seeds_result.index)] = np.array(cost_time_list)
            seeds_result.index = list(np.arange(1, 20 + 1)) + ['Time_Cost']
            seeds_result.columns = methods

            percost_time_result.loc[len(seeds_result.index)] = np.array(cost_time_list)
            percost_time_result.index = list(np.arange(1, 20 + 1)) + ['Time_Cost']
            percost_time_result.columns = methods

            pergbest_list_result.index = list(np.arange(1, 20 + 1))


            with pd.ExcelWriter('./5Hrobustness_result/' + str(H) + dataset[:-4] + '.xlsx') as writer:
                seeds_result.to_excel(writer, sheet_name="Seeds_List", index=True)
                percost_time_result.to_excel(writer, sheet_name="Per_Cost_Time", index=True)
                pergbest_list_result.to_excel(writer, sheet_name="Per_Gbest_List", index=True)


        print('-------------------------Finished-------------------------\n')




