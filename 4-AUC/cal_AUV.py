import numpy as np
import pandas as pd
from scipy.integrate import trapz
import os
from dataload import dataload
from tqdm import tqdm

Datasets = os.listdir('../0-hypergraph_datasets/Ori_data/')
Datasets_name = [x[:-4] for x in Datasets]
# Datasets_name = ['senate-committees']
Algorithms = ['DPSO-HEDV_NG40H1','DPSO_NG40H1','DPSO-HEDV_NG20','DPSO-HEDV_NG5','DPSO-HEDV','DPSO','HEDV-greedy', 'HADP', 'HSDP', 'H-RIS', 'H-CI(I=1)', 'H-CI(I=2)', 'H-Degree', 'Hyper-IMRank']

beta_list = [0.01 ]#,0.02 ,0.015 ,0.005
T_list = [25,]#15,20, 26
AUV_all = pd.DataFrame(columns = Algorithms, index = Datasets_name)
writer = pd.ExcelWriter('D:/MyOneDrive/OneDrive/桌面/Figure/AUV_all_beta_6.xlsx')

for q in range(1): # 四种参数设置
    beta = beta_list[q]
    t    = T_list[q]
    for i, filename in tqdm(enumerate(Datasets_name)):
        # print(filename)
        data = dataload.get_scale(filename, beta)
        x_index_for_k = data.index
        AUV_list = []
        for j, name in enumerate(Algorithms):
            # S = [len(scale_list) for scale_list in data[name]]
            # print(filename, S)
            y_index_for_scale = [scale_list.mean(axis = 0)[t] for scale_list in data[name]]
            auv = trapz(y_index_for_scale, x_index_for_k)
            AUV_list.append(auv)
        AUV_list = np.array(AUV_list)
        AUV_list_normalized = AUV_list/sum(AUV_list)
        # print(AUV_all.loc[:, name])
        AUV_all.loc[filename, :] = np.around(AUV_list_normalized, 4)

    AUV_all.to_excel(writer, sheet_name='beta = %s, t = %s'%(beta,t))

writer.close()
        