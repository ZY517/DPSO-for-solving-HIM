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
    # n = 0datasets_name[0:1]['diseasome.txt', ]
    for dataset in datasets_name[0:1]:  # 文件名
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
        # for H in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]:
        #     # 4、读取已搜索到的种子节点数据
        #     seeds_list = pd.read_excel('C:/Users/ZZY/Desktop/毕业设计/01-Experiments/1-search_seeds/Hrobustness_result/%.1f'%H + dataset[:-4] + '.xlsx', \
        #                                sheet_name="Seeds_List", \
        #                                index_col=0, \
        #                                usecols=[0,1], \
        #                                skipfooter=1)
        #     # seeds_list；行:种子节点数量、列：不同贪婪式方法  元素为不同贪婪式方法求得的对应的种子集
        #     # 5、传播仿真模拟测试验证效果
        #     # MC_influence = simulation_experiments.R_conduct_all_information(dataset, df_hyper_matrix, seeds_list, R, t, beta)
        #     MHPD_influence = simulation_experiments.conduct_MHPD(dataset, df_hyper_matrix, seeds_list, k=25 , beta=0.01)
        #     influence_result = pd.DataFrame(MHPD_influence)
        #     influence_result.index = list(np.arange(1, 11))
        #     influence_result.columns = ['MHPD_influence']
        #     # 将 influence_result 写入 seeds_list 所在的 Excel 文件，新的 sheet 名为 MHDP_influence
        #     with pd.ExcelWriter(
        #             'C:/Users/ZZY/Desktop/毕业设计/01-Experiments/1-search_seeds/Hrobustness_result/%.1f'%H + dataset[
        #                                                                                                         :-4] + '.xlsx',
        #             mode='a', if_sheet_exists='new') as writer:
        #         influence_result.to_excel(writer, sheet_name='MHDP_influence')
        #     # result.to_csv('./beta = %s/'%beta + dataset[:-4] + '_%s.csv'%R, sheet_name="Spread_Scale")
        # print('-------------------------Finished-------------------------\n')
        cols = ['Hypergraph-IMRank','Adeff']
        # 4、读取已搜索到的种子节点数据
        seeds_list = pd.read_excel('D:/MyOneDrive/OneDrive/桌面/01-Experiments/1-search_seeds/HyperIMRankAdeffMIE/' + dataset[:-4] + '.xlsx', \
                                   sheet_name="Seeds_List", \
                                   index_col=0, \
                                   )
        # seeds_list；行:种子节点数量、列：不同贪婪式方法  元素为不同贪婪式方法求得的对应的种子集
        # 5、传播仿真模拟测试验证效果
        # MC_influence = simulation_experiments.R_conduct_all_information(dataset, df_hyper_matrix, seeds_list, R, t, beta)
        MHPD_influence = simulation_experiments.conduct_MHPD(dataset, df_hyper_matrix, seeds_list, k=25 , beta=0.01)
        influence_result = pd.DataFrame(MHPD_influence).T
        influence_result.index = list(np.arange(1, 31))
        influence_result.columns = cols
        # 将 influence_result 写入 seeds_list 所在的 Excel 文件，新的 sheet 名为 MHDP_influence
        output_path = 'D:/MyOneDrive/OneDrive/桌面/01-Experiments/1-search_seeds/HyperIMRankAdeffMIE/' + dataset[:-4] + '.xlsx'

        # 直接写入，不管文件是否存在
        with pd.ExcelWriter(output_path, mode = 'a',if_sheet_exists='new') as writer:
            influence_result.to_excel(writer, sheet_name = 'MHDP_influence')
        # result.to_csv('./beta = %s/'%beta + dataset[:-4] + '_%s.csv'%R, sheet_name="Spread_Scale")
        print('-------------------------Finished-------------------------\n')


        # # 3、设置R的值、模拟跳数t、感染概率beta
        # R = 500  # 仿真实验的次数
        # t = 24
        # beta = 0.01
        # import numpy as np
        # # 4、读取已搜索到的种子节点数据
        # seeds_list = pd.read_excel('D:/毕业设计/01-Experiments/8-experiment/seeds_result/' + dataset[:-4] + '_seed.xlsx', \
        #                            sheet_name="Seeds_List", \
        #                            index_col=0, \
        #                            usecols=[0,1], \
        #                            skipfooter=1)
        # BetaList = []
        # BetaList = np.linspace(0.005, 0.1, 20)
        # print(BetaList)
        # for t in range(1,11):
        #     for beta in BetaList:
        #         print(t,' ',beta)
        #
        #         # seeds_list；行:种子节点数量、列：不同贪婪式方法  元素为不同贪婪式方法求得的对应的种子集
        #         # 5、传播仿真模拟测试验证效果
        #         simulation_experiments.conduct_all_information(dataset, df_hyper_matrix, seeds_list, R, t, beta)
        #         # result.to_csv('./beta = %s/'%beta + dataset[:-4] + '_%s.csv'%R, sheet_name="Spread_Scale")
        #         print('-------------------------Finished-------------------------\n')
