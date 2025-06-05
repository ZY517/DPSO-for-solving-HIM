'''
本程序记载所有搜索算法，可调用后求解影响力最大化问题
'''
import numpy as np
import pandas as pd
import random
import copy
import Hyperspreading
import networkx as nx
from tqdm import tqdm
import time
from Fitness import fitness
import sys
import math


class algorithms:
    # 张子柯文章提出方法：
    ################################################################################################
    # 方法一：度最大化：'Degree'方法
    def degreemax(df_hyper_matrix, K):
        """
        度最大化方法
        """
        begin_time = time.time()
        seed_list_degreemax = []
        degree = algorithms.getTotalAdj(df_hyper_matrix)  # 计算总度数
        for i in tqdm(range(0, K), desc='Degree finished'):  # 对于每个种子节点数量
            seeds = algorithms.getSeeds_sta(degree, i)  # 获取种子节点
            seed_list_degreemax.append(seeds)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_degreemax, cost_time

    # 辅助函数：
    # 1、计算所有节点的度数
    def getTotalAdj(df_hyper_matrix):
        """
        计算所有节点的度数列表：
        度数：与该节点相连接，即在同一个超边下的节点的数量
        """
        deg_list = []  # 初始化度数列表
        N, M = df_hyper_matrix.shape
        nodes_arr = np.arange(N)  # 生成节点索引数组
        for node in nodes_arr:  # 对于每个节点
            node_list = []  # 初始化节点列表
            edge_set = np.where(df_hyper_matrix.loc[node] == 1)[0]  # 找到与节点相连的边集合
            for edge in edge_set:  # 对于每条边
                node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))  # 找到与边相连的节点，并添加到节点列表中
            node_set = np.unique(np.array(node_list))  # 去除重复的节点
            deg_list.append(len(list(node_set)) - 1)  # 计算节点的度数，并添加到度数列表中
        return np.array(deg_list)  # 返回度数列表

    # 2、根据节点度数列表，选择前i个度数最大的节点作为种子集合
    def getSeeds_sta(degree, i):
        """
        根据节点度数选择目标种子集合
        不做其他处理，而是直接选择度数最靠前的几个节点，可能导致影响力重复严重
        """
        # 构建节点和度数矩阵
        matrix = []
        matrix.append(np.arange(len(degree)))
        matrix.append(degree)
        df_matrix = pd.DataFrame(matrix)
        df_matrix.index = ['node_index', 'node_degree']
        # 根据节点度数降序排序
        df_sort_matrix = df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
        # 获取排序后的度数和节点列表
        degree_list = list(df_sort_matrix.loc['node_degree'])
        nodes_list = list(df_sort_matrix.loc['node_index'])
        # 选择前i个节点作为种子节点
        chosed_arr = list(df_sort_matrix.loc['node_index'][:i])
        # 如果度数相同的节点有多个，则随机选择一个
        index = np.where(np.array(degree_list) == degree_list[i])[0]
        nodes_set = list(np.array(nodes_list)[index])
        while 1:
            node = random.sample(nodes_set, 1)[0]
            # 如果该节点不在已选择的节点列表中，则加入选择列表中
            if node not in chosed_arr:
                chosed_arr.append(node)
                break
            else:
                # 如果该节点已经在已选择的节点列表中，则从节点集合中删除，并继续选择
                nodes_set.remove(node)
                continue
        return chosed_arr

    ###################################################################################################
    # 2、方法二：超度最大化：'H-Degree'方法
    def HDegree(df_hyper_matrix, K):
        """
        超度最大化方法
        """
        begin_time = time.time()
        seed_list_HDegree = []
        degree = df_hyper_matrix.sum(axis=1)  # 计算节点的超度数
        for i in tqdm(range(0, K), desc='H-Degree finished'):  # 对于每个种子节点数量
            seeds = algorithms.getSeeds_sta(degree, i)  # 获取种子节点
            seed_list_HDegree.append(seeds)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HDegree, cost_time

    ###################################################################################################
    # 3、方法三：HeuristicDegreeDiscount，基于度折扣的启发式方法
    def HADP(df_hyper_matrix, K):
        """
        HeuristicDegreeDiscount算法是一种基于度数的启发式算法，用于选择种子节点。以下是该算法的伪代码：
        1、初始化一个空的种子节点列表seeds
        2、初始化一个节点度数列表degree，调用函数getTotalAdj(df_hyper_matrix, N)得到每个节点的度数
        3、循环K次：
            a. 找到度数最大的节点，调用函数getMaxDegreeNode(degree, seeds)得到最大度数节点
            b. 将最大度数节点添加到种子节点列表seeds中
            c. 更新度数列表degree，调用函数updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds)
        输出：选择的种子节点列表seeds
        """
        begin_time = time.time()
        seed_list_HUR = []
        seeds = []
        degree = algorithms.getTotalAdj(df_hyper_matrix)
        for j in tqdm(range(1, K + 1), desc="HADP finished"):
            chosenNode = algorithms.getMaxDegreeNode(degree, seeds)
            seeds.append(chosenNode)
            seed_list_HUR.append(seeds.copy())
            algorithms.updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HUR, cost_time

    # 辅助函数：
    # 1、获取当前度数最大的未被选择节点
    def getMaxDegreeNode(degree, seeds):
        """
        获取度数最大的未选择节点
        """
        degree_copy = copy.deepcopy(degree)
        global chosedNode
        while 1:
            flag = 0
            degree_matrix = algorithms.getDegreeList(degree_copy)
            node_index = degree_matrix.loc['node_index']
            for node in node_index:
                if node not in seeds:
                    chosedNode = node
                    flag = 1
                    break
            if flag == 1:
                break
        return chosedNode

    # 2、获取按照度数由大到小重新排序后的节点序列
    def getDegreeList(degree):
        """
        获取按照度数由大到小重新排序后的节点序列
        """
        matrix = []
        matrix.append(np.arange(len(degree)))
        matrix.append(degree)
        df_matrix = pd.DataFrame(matrix)
        df_matrix.index = ['node_index', 'node_degree']
        return df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)

    # 3、使用HUR方法更新节点的度数，主要用于支持HUR节点选择方法
    def updateDeg_hur(degree, chosenNode, df_hyper_matrix, seeds):
        """
        使用HUR方法更新节点的度数，主要用于支持HUR节点选择方法
        """
        edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]  # 获取选择节点所连接的超边索引集合
        adj_set = []
        for edge in edge_set:
            adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))  # 获取与选择节点相连的节点集合
        adj_set_unique = np.unique(np.array(adj_set))  # 去除重复的节点
        for adj in adj_set_unique:  # 遍历与选择节点相连的节点
            adj_edge_set = np.where(df_hyper_matrix.loc[adj] == 1)[0]  # 获取与相连节点相连的超边索引集合
            adj_adj_set = []
            for each in adj_edge_set:
                adj_adj_set.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))  # 获取与相连节点相连的节点集合
            if adj in adj_adj_set:
                adj_adj_set.remove(adj)  # 去除相连节点自身
            sum = 0
            for adj_adj in adj_adj_set:
                if adj_adj in seeds:
                    sum = sum + 1  # 统计相连节点相连的节点中已选择的种子节点的个数
            degree[adj] = degree[adj] - sum  # 更新相连节点的度数

    ###################################################################################################
    # 4、方法四：HeuristicSingleDiscount：简单的度折扣方法
    def HSDP(df_hyper_matrix, K):
        """
        HuresticSingleDiscount algorithm
        """
        begin_time = time.time()
        seed_list_HSD = []
        seeds = []
        degree = algorithms.getTotalAdj(df_hyper_matrix)
        for j in tqdm(range(1, K + 1), desc="HSDP finished"):
            chosenNode = algorithms.getMaxDegreeNode(degree, seeds)
            seeds.append(chosenNode)
            seed_list_HSD.append(seeds.copy())
            algorithms.updateDeg_hsd(degree, chosenNode, df_hyper_matrix)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HSD, cost_time

    # 辅助函数：
    # 1、使用HSD方法更新节点的度数，主要用于支持HSD节点选择方法
    def updateDeg_hsd(degree, chosenNode, df_hyper_matrix):
        """
        使用HSD方法更新节点的度数，主要用于支持HSD点选择方法：
        """
        edge_set = np.where(df_hyper_matrix.loc[chosenNode] == 1)[0]  # 获取选择节点所连接的超边索引集合
        for edge in edge_set:
            node_set = np.where(df_hyper_matrix[edge] == 1)[0]  # 获取与选择节点相连的节点集合
            for node in node_set:
                degree[node] = degree[node] - 1  # 更新相连节点的度数

    ###################################################################################################
    # 5、方法五：'greedy'方法
    def generalGreedy(df_hyper_matrix, K, mtkl=1):
        """
        GeneralGreedy algorithm
        """
        begin_time = time.time()
        degree = df_hyper_matrix.sum(axis=1)
        seed_list_Greedy = []
        seeds = []
        for i in tqdm(range(0, K), desc="General-greedy finished"):
            scale_list_temp = []
            maxNode = 0
            maxScale = 0
            for inode in range(0, len(degree)):
                if inode not in seeds:
                    seeds.append(inode)
                    scale_avg = []
                    for i in range(mtkl):
                        scale_temp, _ = Hyperspreading.Hyperspreading().hyperSI(df_hyper_matrix, seeds)
                        scale_avg.append(scale_temp)
                    scale = np.array(scale_avg).mean()
                    seeds.remove(inode)
                    scale_list_temp.append(scale)
                    if scale > maxScale:
                        maxNode = inode
                        maxScale = scale
            seeds.append(maxNode)
            seed_list_Greedy.append(seeds.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_Greedy, cost_time

    ###################################################################################################
    # 6、方法六：'CI'方法
    def CI(df_hyper_matrix, K, l):
        begin_time = time.time()
        # df_hyper_matrix是超边的邻接矩阵，K是要选择的节点个数，l是计算CI值时的参数，表示使用几阶邻居
        seed_list_CI = []
        seeds = []  # 保存选中的节点
        N, M = df_hyper_matrix.shape  # 获取节点和超边的个数
        n = np.ones(N)  # 初始化一个长度为N的数组，表示节点是否被选中，初始都为1表示未选中
        CI_list = algorithms.computeCI(df_hyper_matrix, l)  # 调用computeCI函数计算所有节点的CI值
        CI_arr = np.array(CI_list)  # 将CI_list转换为numpy数组
        for j in range(0, K):
            # 循环K次，每次选择一个具有最大CI值的节点
            CI_chosed_val = CI_arr[np.where(n == 1)[0]]  # 获取未被选中的节点的CI值
            CI_chosed_index = np.where(n == 1)[0]  # 获取未被选中的节点的索引
            index = np.where(CI_chosed_val == np.max(CI_chosed_val))[0][0]  # 找到最大CI值对应的索引
            node = CI_chosed_index[index]  # 获取对应的节点
            n[node] = 0  # 将选中的节点标记为已选中
            seeds.append(node)  # 将选中的节点添加到seeds列表中
            seed_list_CI.append(seeds.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_CI, cost_time

    # 辅助函数：
    # 1、计算每个节点的CI值
    def computeCI(df_hyper_matrix, l):
        CI_list = []  # 用于存储每个节点的CI值
        degree = df_hyper_matrix.sum(axis=1)  # 每个节点的度数
        N, M = df_hyper_matrix.shape  # 节点和超边的个数
        for i in tqdm(range(0, N), desc="CI (l=%d) finished" % l):  # 遍历每个节点
            edge_set = np.where(df_hyper_matrix.loc[i] == 1)[0]  # 找到与当前节点相连的超边索引
            if l == 1:  # 如果l=1，只考虑一阶邻居
                node_list = []
                for edge in edge_set:
                    node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
                if i in node_list:
                    node_list.remove(i)
                node_set = np.unique(np.array(node_list))  # 找到节点集合，去除重复节点
            elif l == 2:  # 如果l=2，考虑二阶邻居
                node_list = []
                for edge in edge_set:
                    node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
                if i in node_list:
                    node_list.remove(i)
                node_set1 = np.unique(np.array(node_list))  # 找到一阶邻居节点集合
                node_list2 = []
                edge_matrix = np.dot(df_hyper_matrix.T, df_hyper_matrix)
                edge_matrix[np.eye(M, dtype=np.bool_)] = 0
                df_edge_matrix = pd.DataFrame(edge_matrix)
                adj_edge_list = []
                for edge in edge_set:
                    adj_edge_list.extend(list(np.where(df_edge_matrix[edge] != 0)[0]))
                adj_edge_set = np.unique(np.array(adj_edge_list))  # 找到与一阶邻居有共同邻居的超边集合
                for each in adj_edge_set:
                    node_list2.extend(list(np.where(df_hyper_matrix[each] == 1)[0]))
                node_set2 = list(np.unique(np.array(node_list2)))  # 找到二阶邻居节点集合
                for node in node_set2:
                    if node in list(node_set1):  # 去除二阶邻居中已经包含在一阶邻居中的节点
                        node_set2.remove(node)
                node_set = np.array(node_set2)  # 最终得到节点集合
            ki = degree[i]  # 当前节点的度数
            sum = 0
            for u in node_set:  # 遍历节点集合中的每个节点
                sum = sum + (degree[u] - 1)
            CI_i = (ki - 1) * sum  # 计算CI值
            CI_list.append(CI_i)
        return CI_list  # 返回每个节点的CI值列表

    ###################################################################################################
    # 7、方法七：'RIS'方法
    def RIS(df_hyper_matrix, K, lamda, theta):
        begin_time = time.time()
        seed_list_RIS = []
        S = []  # 存储选定的种子节点
        U = []  # 存储每次迭代生成的子图的节点
        N, M = df_hyper_matrix.shape  # 获取节点和超边的个数
        # 迭代theta次
        for theta_iter in tqdm(range(0, theta), desc="RIS finished"):
            df_matrix = copy.deepcopy(df_hyper_matrix)  # 深拷贝超图的邻接矩阵
            # 随机选择一个节点
            selected_node = random.sample(list(np.arange(len(df_hyper_matrix.index.values))), 1)[0]
            # 以1-λ的比例删除边，构成子超图
            all_edges = np.arange(len(df_hyper_matrix.columns.values))  # 所有边的索引
            prob = np.random.random(len(all_edges))  # 随机生成概率
            index = np.where(prob > lamda)[0]  # 概率大于lamda的边的索引
            for edge in index:
                df_matrix[edge] = 0  # 删除边
            # 将子超图映射到普通图
            adj_matrix = np.dot(df_matrix, df_matrix.T)  # 子超图的邻接矩阵
            adj_matrix[np.eye(N, dtype=np.bool_)] = 0  # 将对角线元素置为0
            df_adj_matrix = pd.DataFrame(adj_matrix)
            df_adj_matrix[df_adj_matrix > 0] = 1  # 大于0的元素置为1
            G = nx.from_numpy_array(df_adj_matrix.values)  # 将邻接矩阵转换为图
            shortest_path = nx.shortest_path(G, target=selected_node)  # 得到从随机选择的节点到其他节点的最短路径
            RR = []
            for each in shortest_path:
                RR.append(each)
            U.append(list(np.unique(np.array(RR))))  # 将每次迭代生成的节点加入U
        # 重复K次
        for k in range(0, K):
            U_list = []
            for each in U:
                U_list.extend(each)
            dict = {}  # 存储节点和出现次数的字典
            for each in U_list:
                if each in dict.keys():
                    dict[each] = dict[each] + 1
                else:
                    dict[each] = 1
            candidate_list = sorted(dict.items(), key=lambda item: item[1], reverse=True)  # 按节点出现次数降序排序
            chosed_node = candidate_list[0][0]  # 选择出现次数最多的节点
            S.append(chosed_node)  # 将选定的节点加入S
            seed_list_RIS.append(S.copy())
            for each in U:
                if chosed_node in each:
                    U.remove(each)  # 从U中移除包含选定节点的子图
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_RIS, cost_time

    ###################################################################################################
    # 自提方法
    # 1、方法一：基于HEDV的greedy搜索方法
    def obj_func_greedy(df_hyper_matrix, k, obj_func_name):
        """
        基于目标函数的贪婪策略构建初始解
        """
        begin_time = time.time()
        seed_list_HEDV = []
        obj_func = algorithms.select_obj_func(obj_func_name)
        num_nodes = df_hyper_matrix.shape[0]
        seeds_Greedy = []
        for i in tqdm(range(k), desc='HEDV-greedy'):  # 一共要添加k个节点
            maxNode = 0
            maxfitness = 0
            for inode in range(num_nodes):
                if inode not in seeds_Greedy:
                    seeds_Greedy.append(inode)
                    fitness = obj_func(df_hyper_matrix.values, seeds_Greedy)
                    seeds_Greedy.remove(inode)
                    if fitness > maxfitness:
                        maxNode = inode
                        maxfitness = fitness
            seeds_Greedy.append(maxNode)
            seed_list_HEDV.append(seeds_Greedy.copy())
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_HEDV, cost_time

    # 辅助函数
    # 1、选择目标函数的计算方法
    def select_obj_func(obj_func_name):
        if obj_func_name == 'HEDV':
            return fitness.HEDV
        if obj_func_name == "MC":
            return fitness.MC
        if obj_func_name == 'MHPD':
            return fitness.MHPD
    ###################################################################################################
    # Hyper_IMRank -from Gong Xulu Influence maximization on hypergraphs via multi-hop influence estimation
    def getNeighbour(targetNode, df_hyper_matrix):
        """
        :param targetNode:
        :param df_hyper_matrix:
        :return: the neighbours of the targetNode
        """
        edge_set = np.where(df_hyper_matrix.loc[targetNode] == 1)[0]  # chosenNode所在的Hyperedge集合
        adj_set = []
        for edge in edge_set:
            adj_set.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
        adj_set_unique = np.unique(np.array(adj_set))  # adj_set_unique为chosenNode的邻居集合(包含自身)
        adj_set_unique = list(adj_set_unique)
        if targetNode in adj_set_unique:
            adj_set_unique.remove(targetNode)
        return adj_set_unique  # list

    def getPvw(df_hyper_matrix, N, pev, pve):
        Pvw = np.zeros((N, N), dtype = float)
        factor = 1 - pev * pve
        dic_hyperedge = {}
        dic_nbr = {}
        for inode in range(N):
            dic_nbr[inode] = algorithms.getNeighbour(inode, df_hyper_matrix)
            dic_hyperedge[inode] = np.where(df_hyper_matrix.loc[inode] == 1)[0]  # inode所在的Hyperedge集合

        for inode in range(N):
            inbr = algorithms.getNeighbour(inode, df_hyper_matrix)
            edge_set1 = dic_hyperedge[inode]  # inode所在的Hyperedge集合
            for jnode in inbr:
                edge_set2 = dic_hyperedge[jnode]  # jnode所在的Hyperedge集合
                inter = np.intersect1d(edge_set1, edge_set2)  # 求交集
                num_common_edges = len(inter)
                Pvw[inode][jnode] = 1 - math.pow(factor, num_common_edges)
        return Pvw
    def HyperRankLFA(df_hyper_matrix, N, r, Pvw):
        M = np.ones(N)
        for i in range(N - 1, 0, -1):
            for j in range(i):
                M[r[j]] = M[r[j]] + Pvw[r[j]][r[i]] * M[r[i]]
                M[r[i]] = (1 - Pvw[r[j]][r[i]]) * M[r[i]]

        return M
    def HyperIMRANK(df_hyper_matrix, K):
        """
        The Hyper-IMRANK algorithm
        :param df_hyper_matrix: 节点-超边邻接矩阵
        :param N: 节点数
        :param K: 种子集数
        :return: 种子集列表和时间开销
        """
        seeds_list_HIMR = []
        cost_time = []
        for i in tqdm(range(1,K+1),desc = 'HyperIMRANK'):
            start_time = time.time()
            N = df_hyper_matrix.shape[0]
            pev = 0.01
            pve = 0.01
            Pvw = algorithms.getPvw(df_hyper_matrix, N, pev, pve)
            r_0 = np.arange(N)
            r_1 = r_0
            # M = HyperRankLFA(df_hyper_matrix, N, r_0, Pvw)
            # print(M)
            cnt = 0
            while True:
                cnt += 1
                M = algorithms.HyperRankLFA(df_hyper_matrix, N, r_0, Pvw)
                r_1 = np.argsort(-M)
                if (r_0 == r_1).all():
                    break
                else:
                    r_0 = r_1

            seeds = r_1[0:i]
            seeds = seeds.tolist()
            cur_time = time.time()
            run_time = cur_time - start_time
            seeds_list_HIMR.append(seeds)
            cost_time.append(run_time)
        return seeds_list_HIMR, cost_time

    ###################################################################################################
    # MIE 2/3 -from Gong Xulu Influence maximization on hypergraphs via multi-hop influence estimation






    ###################################################################################################
    # 自提方法
    # 1、方法一：离散粒子群DPSO搜索方法
    def DPSO(df_hyper_matrix, K, obj_func_name, df_c, H,randomseed=55):
        """
        基于目标函数的贪婪策略构建初始解
        c1 c2 自身和群体的学习因子
        r1 r2 随机学习因子
        w     惯性权重
        return seed_list_HEDV, cost_time
               seed_list_HEDV=[[1],[1,2],[1,2,3],[1,2,3,4],....[K个种子]]
        """
        df_adj_matrix = np.dot(df_hyper_matrix, df_hyper_matrix.T)
        # 固定随机种子
        random.seed(randomseed)
        begin_time = time.time()
        num_group = 40  # 粒子群数量    注意：初识粒子群设置的是各不相同，如果排列组合不足 num_group，那么会死循环
        c1 = 2  # 自身学习因子
        c2 = 2  # 群体学习因子
        w = 0.5  # 惯性因子
        iterMax = 10  # 粒子群连续最优无改进次数
        H_value = H
        num_nodes = df_hyper_matrix.shape[0]  # 种群数量
        num_edges = df_hyper_matrix.shape[1]  # 超边数量
        closenode_dict = algorithms.closenode(df_hyper_matrix, df_c) # 节点的紧密邻居节点集合
        obj_func = algorithms.select_obj_func(obj_func_name)
        neibor_matrix = np.dot(df_hyper_matrix, (df_hyper_matrix).T)
        np.fill_diagonal(neibor_matrix, 0)  # 邻接矩阵，元素表示节点ij共有的边数
        # HEDV_list = [obj_func(df_hyper_matrix.values, [i]) for i in range(num_nodes)]  # 节点HEDV排序
        # HEDV_dict = {i:HEDV_list[i] for i in range(num_nodes)}

        GIwCC3_list, neighbors_list = algorithms.GIwCC(num_nodes, df_adj_matrix,df_c)

        HEDV_list = [obj_func(df_hyper_matrix, [i]) for i in range(num_nodes)]
        HEDV_dict = {i:HEDV_list[i] for i in range(num_nodes)}
        # all_real_list = [[]]*K       #保留进化过程的real_best
        seed_list_DPSO = []  # 需要输出的多K值 种子集 集合
        percost_time_list = []  # 记录每次迭代的耗时
        pergbest_list = []
        for k in tqdm(range(1,K+1), desc='DPSO'):  # 一共要添加k个节点

            # 初识种群 ,group
            group = algorithms.initial_neeeds(df_hyper_matrix, num_nodes, num_group, k, HEDV_dict)
            # 初始速度  ,speed_list
            speed_list = algorithms.initial_speeds(num_group, k)
            # 初识种群edv,group_EDV_list
            # group_EDV_list = [obj_func(df_hyper_matrix.values, i) for i in group]
            group_EDV_list = [obj_func(df_hyper_matrix, i) for i in group]
            # 找到PGbest
            pbest, gbest = algorithms.initial_PGbest(group_EDV_list, group)
            # print('pbest:', pbest, '\ngbest', gbest, '\nspeed_list', speed_list)
            # 记录整过过程中的最优粒子（EDV,[种子集]）
            real_best = (max(group_EDV_list), gbest)
            # all_real_list[k-1].append(real_best)
            # 记录历史粒子群体中的最优个体[(第一次迭代),(第二次迭代)]
            gbest_list = [(max(group_EDV_list), gbest)]

            iter = 0
            while iter < iterMax:
                # 更新速度  ,speed_list
                speed_list = [algorithms.renew_speed(speed_list[i], pbest[i], real_best[1], group[i], w, c1, c2, k, H_value) for
                              i in range(num_group)]
                # 更新种群  ,group
                new_group_EDV_list=[]
                for i in range(num_group):
                    seedi, EDVi = algorithms.renew_seed(df_hyper_matrix,group[i], speed_list[i],group_EDV_list[i], k, num_nodes, HEDV_dict,"Random", closenode_dict, obj_func)
                    group[i] = seedi
                    new_group_EDV_list.append(EDVi)
                # 更新PGbest
                pbest, gbest, gbest_edv = algorithms.renew_best(df_hyper_matrix, pbest, obj_func, group_EDV_list,
                                                                new_group_EDV_list, group, neibor_matrix, num_group, k,
                                                                num_nodes)

                group_EDV_list = new_group_EDV_list.copy()  #旧EDV存档
                # 局部搜索
                # gbest, gbest_edv = algorithms.local_search(df_hyper_matrix, obj_func, gbest, neibor_matrix, k, num_nodes)
                gbest_list.append((gbest_edv, gbest))

                if gbest_edv > real_best[0]:
                    real_best = (gbest_edv, gbest)
                    iter = 0
                else:
                    iter += 1
                # all_real_list[k - 1].append(real_best)
                # print(iter)

            seed_list_DPSO.append(real_best[1])
            percost_time_list.append(time.time()-begin_time)
            pergbest_list.append(gbest_list)
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_DPSO, cost_time , pergbest_list , percost_time_list

    # 辅助函数
    # 1、选择目标函数的计算方法

    # 考虑集聚系数的节点全局影响力Considering the Global Influence of Nodes with Agglomeration Coefficient
    def GIwCC(num_nodes,df_adj_matrix,df_c):
        GIwCC1_list = []
        GIwCC2_list = []
        GIwCC3_list = []
        neighbors_list=[]
        for i in range(num_nodes):
            neibors = Hyperspreading.Hyperspreading.getNodesofHpe(i, df_adj_matrix)
            GIwCC1_list.append(len(neibors))
            neighbors_list.append(neibors)
        for i in range(num_nodes):
            GIwCC2_list = GIwCC1_list.copy()
            for j in neighbors_list[i]:
                GIwCC2_list[i] += GIwCC1_list[j]*df_c
        for i in range(num_nodes):
            CIwCC3_list = GIwCC1_list.copy()
            for j in neighbors_list[i]:
                CIwCC3_list[i] += GIwCC2_list[j]*df_c

        return GIwCC3_list, neighbors_list
    # 2、初始化粒子群
    def initial_neeeds(df_hyper_matrix, num_nodes, num_group, k, HEDV_dict):
        sort_HEDV_dict = sorted(HEDV_dict.items(), key=lambda item: item[1], reverse=True)
        # sort_HEDV_dict = [(468, 1.1), (469, 1.1), (471, 1.1), (472, 1.1), (383, 1.09), (385, 1.09), (386, 1.09),......]
        need_best = [sort_HEDV_dict[i][0] for i in range(k)]
        need_worst = [sort_HEDV_dict[i][0] for i in range(k, num_nodes)]
        # print(need_best)
        # print(need_worst)
        group = [need_best]
        count = 1
        while count < num_group:
            needi = need_best.copy()
            needj = need_worst.copy()
            for i in range(k):
                if random.random() > 0.5:
                    needi[i] = random.choice(needj)
                    needj.remove(needi[i])
            if needi in group:
                continue
            else:
                group.append(needi)
                count += 1
            # print(count)
        return group

    # 3、初始化速度
    # def initial_speeds(num_group,k):
    #     speed_list = []
    #     for i in range(num_group):
    #         speedi = []
    #         for j in range(k):
    #             speedi.append(0)
    #         speed_list.append(speedi)
    #     return speed_list
    def initial_speeds(num_group, k):
        v_i = [0] * k
        V_list = [v_i for i in range(num_group)]
        return V_list

    # 3.5 定义适应度计算
    # Fitness

    # 4、初始化Pbest、Gbest
    def initial_PGbest(group_EDV_list, group):
        # print("初识种群edv: ",group_EDV_list)
        gbest = group[group_EDV_list.index(max(group_EDV_list))]
        pbest = group.copy()
        return pbest, gbest

    # 5、更新速度
    def renew_speed(v_i, pbest_i, gbest_i, needs_i, w, c1, c2, k, H_value):
        r1 = random.random()
        r2 = random.random()
        v_j = (w * np.array(v_i) + \
               c1 * r1 * np.array(algorithms.position_AND_position(pbest_i, needs_i, k)) + \
               c2 * r2 * np.array(algorithms.position_AND_position(gbest_i, needs_i, k)))
        # print('没有归整的V_j:  ',v_j)
        return algorithms.H(v_j, k, H_value)

    def position_AND_position(best_i, needs_i, k):
        v = [0] * k
        for i in range(k):
            if needs_i[i] not in best_i:
                v[i] = 1
        return (v)

    def H(v_i, k, H_value):
        for i in range(k):
            if v_i[i] >= H_value:
                v_i[i] = 1
            else:
                v_i[i] = 0
        return v_i

    # 6、更新粒子
    def renew_seed(df_hyper_matrix,needs_i, v_i,EDV_i, k, num_nodes, HEDV_dict, heuristic , closenode_dict, obj_func):
        nodes = list(range(num_nodes))
        candid_nodes = set(nodes)
        # 找出需要替换的位置 rs_list
        rs_list = [rs for rs in range(k) if v_i[rs] == 1]
        if len(rs_list) == 0:
            return needs_i, EDV_i
        # 排除 close nodes 和当前 seeds
        for node in needs_i:
            candid_nodes -= set(closenode_dict.get(node, []))  # 排除紧密连接的节点
        candid_nodes -= set(needs_i)  # 排除当前 seeds

        candid_nodes = list(candid_nodes)

        probs = None
        if heuristic == 'HEDV':
            weights = [HEDV_dict[c] for c in candid_nodes]
            sum_weights = sum(weights)
            if sum_weights == 0:
                probs = None  # 均匀分布
                selected_nodes = random.sample(candid_nodes, (len(rs_list)))
            else:
                probs = [w / sum_weights for w in weights]
                selected_nodes = random.sample(candid_nodes, weights=probs, k=(len(rs_list)))
        else:
            probs = None  # 随机选择
            selected_nodes = random.sample(candid_nodes, (len(rs_list)))

        i = 0
        while i < 10:  # 最多尝试10次
            new_needs_i = needs_i.copy()
            if probs == None:
                selected_nodes = random.sample(candid_nodes, (len(rs_list)))
            else:
                selected_nodes = random.sample(candid_nodes, weights = probs, k = (len(rs_list)))

            for idx in range(len(rs_list)):
                new_needs_i[rs_list[idx]] = selected_nodes[idx]

            new_EDV_i = obj_func(df_hyper_matrix, new_needs_i )
            if new_EDV_i > EDV_i:
                return new_needs_i, new_EDV_i
            else:
                i += 1
        return needs_i, EDV_i


    def replace(needs_i, HEDV_dict,heuristic, candid_nodes, rs_list):
        if heuristic =='HEDV':
            sumHEDV = sum([HEDV_dict[i] for i in candid_nodes])
            candid_weights = [HEDV_dict[i]/sumHEDV for  i in candid_nodes]

            selected_candid = random.choices(candid_nodes, candid_weights)[0]
        else:
            selected_candid = random.choices(candid_nodes)[0]
        return selected_candid

    # 紧密节点
    def closenode(df_hyper_matrix, df_c):
        """
        计算与每个节点链接最紧密的 df_c 个节点。

        参数:
            df_hyper_matrix (pd.DataFrame): 节点-超边邻接矩阵 (n_nodes x n_edges)，值为 0 或 1。
            df_c (int): 每个节点需要返回的最紧密连接的邻居节点数。

        返回:
            dict: 键是节点编号，值是与该节点链接最紧密的 df_c 个节点列表（按紧密程度降序排列）。
        """
        # 确保输入是数值型数组
        if isinstance(df_hyper_matrix, pd.DataFrame):
            matrix = df_hyper_matrix.values.astype(int)
        else:
            matrix = np.array(df_hyper_matrix, dtype = int)

        # 计算节点之间的共现矩阵 (共处超边的数量)
        co_occurrence_matrix = np.dot(matrix, matrix.T)
        n = co_occurrence_matrix.shape[0]

        # 初始化结果字典
        close_nodes_dict = {}

        # 遍历每个节点
        for node in range(n):
            # 获取该节点与其他节点的共现次数
            co_occurrence = co_occurrence_matrix[node, :].copy()

            # 排除自身
            co_occurrence[node] = -1

            # 处理共现次数相同的情况：优先选择编号较小的节点
            # 使用稳定的排序算法确保可预测的行为
            # 获取所有非零节点的索引和值
            non_zero_indices = []
            non_zero_values = []
            for idx, count in enumerate(co_occurrence):
                if count > 0:
                    non_zero_indices.append(idx)
                    non_zero_values.append(count)

            # 按共现次数降序排序非零节点
            sorted_indices = [x for _, x in sorted(zip(non_zero_values, non_zero_indices), reverse = True)]

            # 选择前k个非零节点
            k = min(len(sorted_indices), df_c)
            selected = sorted_indices[:k]

            # 如果不足c个，随机补充剩余节点
            if k < df_c:
                # 获取所有节点索引（包括共现次数为0的）
                all_indices = list(range(n))
                # 排除已选择的节点
                remaining_indices = [idx for idx in all_indices if idx not in selected]
                # 随机选择补充节点
                additional = random.sample(remaining_indices, df_c - k)
                selected.extend(additional)
            closest_nodes = np.argsort(-co_occurrence, kind = 'stable')[:df_c]

            # # 过滤掉无效节点（共现次数为0或负的）
            # valid_nodes = [j for j in closest_nodes if co_occurrence[j] > 0]

            # # 将结果存入字典
            # close_nodes_dict[node] = valid_nodes

        return close_nodes_dict

    # 7.5 定义局部搜索
    def local_search(df_hyper_matrix, obj_func, need, neibor_matrix, k, num_nodes):
        cur_EDV = obj_func(df_hyper_matrix.values, need)
        for i in range(k):
            needi = need.copy()
            i_neibor = []
            for k in range(num_nodes):
                if neibor_matrix[k][need[i]] > 0:
                    i_neibor.append(k)
            for j in i_neibor:
                if j in need:
                    continue
                else:
                    needi[i] = j
                    i_EDV = obj_func(df_hyper_matrix.values, needi)
                    if i_EDV > cur_EDV:
                        need[i] = j
                        cur_EDV = i_EDV
        return need, cur_EDV

    # 7、更新Pbest、Gbest
    def renew_best(df_hyper_matrix, pbest, obj_func, group_EDV_list, new_group_EDV_list, new_group, neibor_matrix,
                   num_group, k, num_nodes):  # return pbest,gbest,gbest_edv

        for i in range(num_group):
            if new_group_EDV_list[i] > group_EDV_list[i]:
                pbest[i] = new_group[i]

        new_gbest_edv = max(new_group_EDV_list)
        gbest = new_group[new_group_EDV_list.index(new_gbest_edv)]

        return pbest, gbest, new_gbest_edv

    # 计算所有需要替换掉的节点集合的  需要删掉的节点
    def candidate_nodes(df_hyper_matrix, seeds):
        deleted_nodes = []




        return deleted_nodes
    ####################临时——可视化DPSO进化###################################
    def DPSO_K(df_hyper_matrix, K, obj_func_name):
        """
        只记录K规模的进化历程
        基于目标函数的贪婪策略构建初始解
        c1 c2 自身和群体的学习因子
        r1 r2 随机学习因子
        w     惯性权重
        return seed_list_HEDV, cost_time
               seed_list_HEDV=[[1],[1,2],[1,2,3],[1,2,3,4],....[K个种子]]
        """
        begin_time = time.time()
        num_group = 100  # 粒子群数量
        c1 = 2  # 自身学习因子
        c2 = 2  # 群体学习因子
        w = 0.5  # 惯性因子
        iterMax = 10  # 粒子群连续最优无改进次数
        num_nodes = df_hyper_matrix.shape[0]  # 种群数量
        num_edges = df_hyper_matrix.shape[1]  # 超边数量
        obj_func = algorithms.select_obj_func(obj_func_name)
        neibor_matrix = np.dot(df_hyper_matrix, (df_hyper_matrix).T)
        np.fill_diagonal(neibor_matrix, 0)  # 邻接矩阵，元素表示节点ij共有的边数
        HEDV_list = [obj_func(df_hyper_matrix, [i]) for i in range(num_nodes)]  # 节点HEDV排序
        HEDV_dict = {i:HEDV_list[i] for i in range(num_nodes)}
        all_real_list = []       #保留进化过程的real_best
        seed_list_DPSO = []  # 需要输出的多K值 种子集 集合
        for k in tqdm(range(K, K + 1), desc='DPSO'):  # 一共要添加k个节点

            # 初识种群 ,group
            group = algorithms.initial_neeeds(df_hyper_matrix, num_nodes, num_group, k, HEDV_dict)
            print(group)
            # 初始速度  ,speed_list
            speed_list = algorithms.initial_speeds(num_group, k)
            # 初识种群edv,group_EDV_list
            group_EDV_list = [obj_func(df_hyper_matrix, i) for i in group]
            # 找到PGbest
            pbest, gbest = algorithms.initial_PGbest(group_EDV_list, group)
            # print('pbest:', pbest, '\ngbest', gbest, '\nspeed_list', speed_list)
            # 记录整过过程中的最优粒子（EDV,[种子集]）
            real_best = (max(group_EDV_list), gbest)
            all_real_list.append(real_best)
            # 记录历史粒子群体中的最优个体[(第一次迭代),(第二次迭代)]
            gbest_list = [(max(group_EDV_list), gbest)]

            iter = 0
            while iter < iterMax:
                # 更新速度  ,speed_list
                speed_list = [algorithms.renew_speed(speed_list[i], pbest[i], real_best[1], group[i], w, c1, c2, k) for
                              i in range(num_group)]
                # 更新种群  ,group
                group = [algorithms.renew_seed(group[i], speed_list[i], k, num_nodes, HEDV_dict,"HEDV") for i in
                         range(num_group)]
                # 更新PGbest
                new_group_EDV_list = [obj_func(df_hyper_matrix, i) for i in group]
                pbest, gbest, gbest_edv = algorithms.renew_best(df_hyper_matrix, pbest, obj_func, group_EDV_list,
                                                                new_group_EDV_list, group, neibor_matrix, num_group, k,
                                                                num_nodes)
                # 局部搜索
                # gbest, gbest_edv = algorithms.local_search(df_hyper_matrix, obj_func, gbest, neibor_matrix, k, num_nodes)
                gbest_list.append((gbest_edv, gbest))

                if gbest_edv > real_best[0]:
                    real_best = (gbest_edv, gbest)
                    iter = 0
                else:
                    iter += 1
                all_real_list.append(real_best)
                # print(iter)

            seed_list_DPSO.append(real_best[1])
        end_time = time.time()
        cost_time = end_time - begin_time
        return seed_list_DPSO, cost_time,  gbest_list


def DPSO_MHPD(df_hyper_matrix, K, obj_func_name):
    """
    只记录K规模的进化历程
    基于目标函数的贪婪策略构建初始解
    c1 c2 自身和群体的学习因子
    r1 r2 随机学习因子
    w     惯性权重
    return seed_list_HEDV, cost_time
           seed_list_HEDV=[[1],[1,2],[1,2,3],[1,2,3,4],....[K个种子]]
    """
    begin_time = time.time()
    num_group = 100  # 粒子群数量
    c1 = 2  # 自身学习因子
    c2 = 2  # 群体学习因子
    w = 0.5  # 惯性因子
    iterMax = 10  # 粒子群连续最优无改进次数
    num_nodes = df_hyper_matrix.shape[0]  # 种群数量
    num_edges = df_hyper_matrix.shape[1]  # 超边数量
    obj_func = algorithms.select_obj_func(obj_func_name)
    neibor_matrix = np.dot(df_hyper_matrix, (df_hyper_matrix).T)
    np.fill_diagonal(neibor_matrix, 0)  # 邻接矩阵，元素表示节点ij共有的边数
    HEDV_list = [obj_func(df_hyper_matrix, [i]) for i in range(num_nodes)]  # 节点HEDV排序
    HEDV_dict = {i: HEDV_list[i] for i in range(num_nodes)}
    all_real_list = []  # 保留进化过程的real_best
    seed_list_DPSO = []  # 需要输出的多K值 种子集 集合
    for k in tqdm(range(K, K + 1), desc='DPSO'):  # 一共要添加k个节点

        # 初识种群 ,group
        group = algorithms.initial_neeeds(df_hyper_matrix, num_nodes, num_group, k, HEDV_dict)
        print(group)
        # 初始速度  ,speed_list
        speed_list = algorithms.initial_speeds(num_group, k)
        # 初识种群edv,group_EDV_list
        group_EDV_list = [obj_func(df_hyper_matrix, i) for i in group]
        # 找到PGbest
        pbest, gbest = algorithms.initial_PGbest(group_EDV_list, group)
        # print('pbest:', pbest, '\ngbest', gbest, '\nspeed_list', speed_list)
        # 记录整过过程中的最优粒子（EDV,[种子集]）
        real_best = (max(group_EDV_list), gbest)
        all_real_list.append(real_best)
        # 记录历史粒子群体中的最优个体[(第一次迭代),(第二次迭代)]
        gbest_list = [(max(group_EDV_list), gbest)]

        iter = 0
        while iter < iterMax:
            # 更新速度  ,speed_list
            speed_list = [algorithms.renew_speed(speed_list[i], pbest[i], real_best[1], group[i], w, c1, c2, k) for
                          i in range(num_group)]
            # 更新种群  ,group
            group = [algorithms.renew_seed(group[i], speed_list[i], k, num_nodes, HEDV_dict, "HEDV") for i in
                     range(num_group)]
            # 更新PGbest
            new_group_EDV_list = [obj_func(df_hyper_matrix, i) for i in group]
            pbest, gbest, gbest_edv = algorithms.renew_best(df_hyper_matrix, pbest, obj_func, group_EDV_list,
                                                            new_group_EDV_list, group, neibor_matrix, num_group, k,
                                                            num_nodes)
            # 局部搜索
            # gbest, gbest_edv = algorithms.local_search(df_hyper_matrix, obj_func, gbest, neibor_matrix, k, num_nodes)
            gbest_list.append((gbest_edv, gbest))

            if gbest_edv > real_best[0]:
                real_best = (gbest_edv, gbest)
                iter = 0
            else:
                iter += 1
            all_real_list.append(real_best)
            # print(iter)

        seed_list_DPSO.append(real_best[1])
    end_time = time.time()
    cost_time = end_time - begin_time
    return seed_list_DPSO, cost_time, gbest_list