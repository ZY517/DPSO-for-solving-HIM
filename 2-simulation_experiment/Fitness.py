'''
用于估计选定种子的好坏
'''
import numpy as np
from Hyperspreading import Hyperspreading
class fitness:
    # 方法一：HEDV, Hypergragh EDV
    def HEDV(df_hyper_matrix, seed_set, beta=0.01):
        # 只考虑一跳范围内传播节点数量的期望
        # 1、首先确定初始种子集的一跳区域（除开种子外的）
        one_hop_nodes = fitness.get_one_hop_Nodes_of_Nodes(seed_set, df_hyper_matrix) # 获取所有一跳区域节点
        one_hop_nodes_set = [x for x in one_hop_nodes if x not in seed_set] # 除去种子节点部分
        
        # 2、遍历每一个邻居节点i，计算其被感染的概率，求和
        EDV = len(seed_set)
        for i,inode in enumerate(one_hop_nodes_set):
            adj_inodes_list = fitness.get_one_hop_Nodes_of_Node(inode, df_hyper_matrix) # 该邻居节点的邻居
            seeds_involved = [x for x in adj_inodes_list if x in seed_set] # 与该节点相连的种子节点
            pro_not_i = 1
            for j in seeds_involved:
                # 两个点之间的传染概率并不是beta，而是什么呢？？？beta*某个种子节点与该节点共有的超边数/这个种子节点自身所有的超边数
                pro_not_i *= (1-beta*fitness.cal_select_probability(inode, j, df_hyper_matrix))
            pro_i = 1-pro_not_i
            EDV += pro_i
        return EDV
     
    # 辅助函数：
    def getHpe(inode, matrix):
        """
        获取给定节点所在的超边列表
        """
        return np.where(matrix[inode, :] == 1)[0]


    def getNodesofHpe(hpe, matrix):
        """
        获取给定超边中的所有节点列表
        """
        return np.where(matrix[:, hpe] == 1)[0]
    
    def getNodesofHpeSet(hpe_set, matrix):
        """
        获取超边集合中涉及到的所有节点
        """
        adj_nodes = []
        for hpe in hpe_set:
            adj_nodes.extend(fitness.getNodesofHpe(hpe, matrix))
        return np.array(adj_nodes)
    
    def get_one_hop_Nodes_of_Node(node, matrix):
        """
        获取节点集合的所有一跳区域节点
        """
        edge_set = fitness.getHpe(node, matrix) # 获得该节点所在的所有超边
        one_hop_nodes = fitness.getNodesofHpeSet(edge_set, matrix) # 获得超边集中所有节点
        one_hop_nodes_set = fitness.drop_duplicates(one_hop_nodes) # 去除重复
        return one_hop_nodes_set
    
    def get_one_hop_Nodes_of_Nodes(nodes, matrix):
        """
        获取节点集合的所有一跳区域节点
        """
        one_hop_nodes_set = [] # 初始化
        for inode in nodes:
            edge_set = fitness.getHpe(inode, matrix) # 获得该节点所在的所有超边
            one_hop_nodes = fitness.getNodesofHpeSet(edge_set, matrix) # 获得超边集中所有节点
            one_hop_nodes_set.extend(one_hop_nodes) # 将得到的所有节点添加到列表中
        one_hop_nodes_set = fitness.drop_duplicates(one_hop_nodes_set) # 去除重复
        return one_hop_nodes_set

    def drop_duplicates(nodes_list):
        """
        去除列表重复
        """
        nodes_list = list(set(nodes_list)) # 去重
        return nodes_list

    def cal_select_probability(adj_node, seed, df_hyper_matrix):
        j_edges = fitness.getHpe(seed, df_hyper_matrix)
        j_num_edges = len(j_edges)
        if j_num_edges == 0:
            return 0
        else:
            j_include_i_num_edges = 0
            for edge in j_edges:
                if adj_node in fitness.getNodesofHpe(edge, df_hyper_matrix):
                    j_include_i_num_edges += 1
            return j_include_i_num_edges/j_num_edges                     #相邻次数*随机选择感染一条超边的概率

    # 方法二：蒙塔卡洛模拟
    def MC(df_hyper_matrix, seed_set):
        mc_list=[]
        for i in range(200):
            _,I_list = Hyperspreading().hyperSI(df_hyper_matrix, seed_set)
            mc_list.append(len(I_list))
        mc_value=sum(mc_list)/len(mc_list)
        return mc_value

    # 方法三：MHPD

    def Adjacency_matrix_RP(df_hyper_matrix):
        # 1、根据节点超图矩阵获得RP模式下的节点之间的邻接矩阵
        # 其中每个元素代表两个节点之间的共有超边数
        m, n = df_hyper_matrix.shape
        Matrix = np.zeros((m, m))
        for i in range(m):
            i_neighbors = []
            i_edges = np.where(df_hyper_matrix.values[i, :] == 1)[0]
            for edge in i_edges:
                nodes_of_edge = np.where(df_hyper_matrix.values[:, edge] == 1)[0]
                i_neighbors.extend([x for x in nodes_of_edge if x != i])
            for j in range(m):
                if i == j:
                    Matrix[i, j] = np.nan
                else:
                    Matrix[i, j] = i_neighbors.count(j)
        return Matrix

    def Adjacency_matrix_CP(df_hyper_matrix):
        # 1、根据节点超图矩阵获得CP模式下的节点之间的邻接矩阵
        # 其中每个元素代表共有超边数除以该节点自己的超度
        m, n = df_hyper_matrix.shape
        RP_matrix = fitness.Adjacency_matrix_RP(df_hyper_matrix)
        HEdges = [len(np.where(df_hyper_matrix.values[node, :] == 1)[0]) for node in range(m)]
        Matrix = RP_matrix
        for i in range(m):
            if HEdges[i] > 0:
                Matrix[i, :] /= HEdges[i]
        return Matrix

    def MHPD(hyper_matrix, seeds, k=25 , beta=0.01 , model='CP'):
        """
        输入：
        超图、种子节点集合、评估阶段数量k、节点之间传播概率beta、传播模型可选CP或RP
        """
        infect_matrix = fitness.cal_infect_matrix(model, hyper_matrix, beta)
        m, n = hyper_matrix.shape
        # spread_scale = []
        nodes_prob = []  # 存储不同k情况下节点被感染的概率
        # 根据传播模型和超图结构确定节点之间在一次传播过程中的感染概率

        nodes_data = np.zeros(m)  # 存储节点当前被感染的概率
        for i in seeds:
            nodes_data[i] = 1  # 种子的data为1
        nodes_prob.append(nodes_data.copy())  # 保存初始数据

        for i in range(k):  # 一共要更新k次
            # print(nodes_data-1)

            nodes_data = fitness.nodes_prob_update(infect_matrix, nodes_data)
            nodes_prob.append(nodes_data.copy())

        nodes_prob_k = nodes_prob[-1]  # 返回最后一次的节点状态列表

        return sum(nodes_prob_k)

    def cal_infect_matrix(model, hyper_matrix, beta):
        if model == 'RP':
            infect_matrix = 1 - (1 - beta) ** fitness.Adjacency_matrix_RP(hyper_matrix)
        elif model == 'CP':
            infect_matrix = fitness.Adjacency_matrix_CP(hyper_matrix) * beta
        return infect_matrix

    def nodes_prob_update(infect_matrix, nodes_data):
        raw_nodes_data = nodes_data.copy()  # 备份，用这个来计算
        for i in range(len(nodes_data)):  # 第i个节点，非种子节点才需要更新
            raw_prob = raw_nodes_data[i]
            neighbors = np.where(infect_matrix[i, :] > 0)[0]
            new_p_list_infect = raw_nodes_data[neighbors]
            new_p_list_spread = np.array([infect_matrix[x, i] for x in neighbors])
            new_p_list = list(new_p_list_infect * new_p_list_spread)
            nodes_data[i] =  fitness.prob_meanwhile(raw_prob, new_p_list)
        return nodes_data

    def prob_meanwhile(raw_prob, new_p_list):
        # 输入一串感染概率P_list输出整体感染概率
        new_p_list.append(raw_prob)
        result = 1 - np.prod(1 - np.array(new_p_list))
        return result