# 用于加载数据集

import pandas as pd
import numpy as np

class HyperGraph:
    node_dict = {}
    HG = {}      #创建一部字典，用于存储  {边:节点集}
    # 1、设置初始化方法，实例化类时需要传入目标数据的path
    def __init__(self,path):
        self.path = path
    # 2、读取数据集中的数据，将结果存储在类中
    def dataload(self):
        # 2.1 读取数据，获得节点列表字典
        df = pd.read_csv(self.path, index_col=False, header=None) # 读取指定路径的CSV文件，不指定列名和索引列
        arr = df.values    # 将DataFrame转换为二维数组
        pro_arr=[]
        node_list = [] # 创建一个空列表，用于存储节点
        for each in arr:
            node_list.extend(list(map(int, each[0].split(" ")))) # 将每行的字符串转换为整数列表，并添加到node_list列表中
            pro_arr.append(list(map(int, each[0].split(" "))))
        node_arr = np.unique(np.array(node_list)) # 使用numpy库的unique方法去除重复的节点，并转换为数组
        # 遍历node_arr数组的每个元素的索引和值
        self.num_node = len(list(node_arr))  # 数据中涉及的节点的数量
        self.num_edge = len(arr)
        for i in range(self.num_node):
            self.node_dict[node_arr[i]] = i # 将节点和对应的索引添加到node_dict字典中   {节点:节点的索引号}
        for i in range(self.num_edge):
            self.HG[i]=[self.node_dict[k] for k in pro_arr[i]]
        # 2.2 读取数据，将数据转换成超边矩阵//获取关联矩阵、邻居矩阵、点映射边集、边映射点集
        relevance_matrix = np.zeros((self.num_node, self.num_edge))  # 关联矩阵，表示节点i边j的关系
        for i in range(self.num_edge):
            x = self.HG[i]
            for j in x:
                relevance_matrix[j][i] = 1

        neibor_matrix = np.dot(relevance_matrix, (relevance_matrix).T)  # 邻接矩阵，元素表示节点ij共有的超边数,对角线元素为点超度
        dianchaodu=list((np.diag(neibor_matrix)).copy())
        for i in range(self.num_node):
            neibor_matrix[i][i]-=dianchaodu[i]

        node_edges_dict = {}  # 节点所在的所有边         直接调用，不用每次都要遍历寻找
        for i in range(self.num_node):
            node_edges_dict[i] = (np.where(relevance_matrix[i, :] > 0))[0]

        edge_nodes_dict = {}
        for i in range(self.num_edge):
            edge_nodes_dict[i] = (np.where(relevance_matrix[:, i] > 0))[0]

        dianchaodu_list = {}  # 点超度，包含节点的超边数
        for i in range(self.num_node):
            dianchaodu_list[i] = dianchaodu[i]

        dianqiangdu_list = {}  #点强度，考虑邻居节点重复次数
        for i in range(self.num_node):
            dianqiangdu_list[i] = np.sum(neibor_matrix[i, :])

        diandu_list = {}  # 点度，没有重复算上不同超边中的相同相邻节点
        for i in range(self.num_node):
            diandu_list[i] = len(np.where(neibor_matrix[i, :] > 0)[0])

        HE_list = {} #超边基数，每个超边所包含的节点数量
        for i in range(self.num_edge):
            HE_list[i] = len(self.HG[i])




        self.relevance_matrix = relevance_matrix
        self.neibor_matrix = neibor_matrix
        self.node_edges_dict = node_edges_dict
        self.edge_nodes_dict = edge_nodes_dict
        self.diandu_list = diandu_list
        self.dianchaodu_list = dianchaodu_list
        self.dianqiangdu_list = dianqiangdu_list
        self.HE_list = HE_list

