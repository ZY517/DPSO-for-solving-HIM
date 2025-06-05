from Dataloader import dataloader
from Simulation_Experiments import simulation_experiments
import os
import pandas as pd
import time
import numpy as np
# # 设置等待的时间（秒）
# wait_seconds = 3*3600
#
# # 等待指定的时间
# time.sleep(wait_seconds)
# print('a')
if __name__ == '__main__':
    # 1、获取路径下面所有数据集的名称
    datasets_name = os.listdir('../0-hypergraph_datasets/Ori_data/')

    # 2、依次读取这些文件，并对每一个数据使用不同算法进行搜索
    # n = 0datasets_name[0:1] ['senate-committees.txt', ]
    for dataset in datasets_name[4:5]:  # 文件名
        path = '../0-hypergraph_datasets/Ori_data/' + dataset
        print('-------------------------Simulation Experiment--%s-------------------------' % (dataset))
        # 2、调用dataloader类的dataload方法，获取必要信息
        dl = dataloader(path)
        dl.dataload()
        df_hyper_matrix = dl.hyper_matrix  # 从数据中获取的节点超边矩阵

        # 3、设置R的值、模拟跳数t、感染概率beta
        R = 2000  # 仿真实验的次数
        t = 30
        beta = 0.01

        # 4、读取已搜索到的种子节点数据
        # 6.3注释
        # seeds_list = pd.read_excel('C:/Users/ZZY/Desktop/毕业设计/01-Experiments/1-search_seeds/random_result_NG40H1/' + dataset[:-4] + '.xlsx', \
        #                            sheet_name="Seeds_List", \
        #                            index_col=0, \
        #                            usecols=[0,1], \
        #                            skipfooter=1)

        seeds_list = pd.read_excel(
            'D:/MyOneDrive/OneDrive/桌面/01-Experiments/1-search_seeds/HyperIMRankAdeffMIE/' + dataset[
                                                                                                  :-4] + '.xlsx', \
            sheet_name = "Seeds_List", \
            index_col = 0, \
            skipfooter = 0)

        # seeds_list；行:种子节点数量、列：不同贪婪式方法  元素为不同贪婪式方法求得的对应的种子集
        # 5、传播仿真模拟测试验证效果
        simulation_experiments.conduct_all_information(dataset, df_hyper_matrix, seeds_list, R, t, beta)

        print('-------------------------Finished-------------------------\n')

