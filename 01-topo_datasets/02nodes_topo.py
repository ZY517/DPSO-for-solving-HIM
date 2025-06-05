import numpy as np
import pandas as pd
from _ImportHyperGraph import HyperGraph
import os
from tqdm import tqdm
from Hyperspreading import Hyperspreading
from Fitness import fitness

def Basic_topo(filepath):

    H = HyperGraph(filepath)
    H.dataload()

    dianqiangdu_list = H.dianqiangdu_list
    dianchaodu_list = H.dianchaodu_list
    diandu_list = H.diandu_list

    df = pd.DataFrame()

    df['deg'] = diandu_list
    df['d^H'] = dianchaodu_list
    df['d^S'] = dianqiangdu_list

    return df


filepath = 'D:/毕业设计/01-Experiments/0-hypergraph_datasets/'
# file_name = os.listdir(filepath)
file_name = ['iJO1366.txt']
out_path = 'D:/毕业设计/01-Experiments/01-topo_datasets/nodes_topo_result/'

for i in file_name:
    out_file_path = out_path + i[:-4] + '.xlsx'
    Basic_topo(filepath+i).to_excel(out_file_path)