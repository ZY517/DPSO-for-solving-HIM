import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Dataloader import dataloader
import os


'''# 定义要拟合的函数形式'''

# 线性函数
def linear_func(x, a, b):
    return a * x + b


# 多项式函数
def polynomial_func(x, *coefficients):
    return np.polyval(coefficients, x)


# 指数函数
def exponential_func(x, a, b):
    return a * np.exp(b * x)


# 幂函数
def power_func(x, a, b):
    return a * np.power(x, b)


# 对数函数
def logarithmic_func(x, a, b):
    return a * np.log(x) + b


# 计算均方根误差（RMSE）
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# 计算调整R平方
def calculate_adjusted_r_squared(y_true, y_pred, num_params):
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (sse / sst)
    adjusted_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - num_params - 1+0.00000000001))
    return adjusted_r_squared

def Division(values, counts, divisor):   #用于将横坐标划分成指定大小（divisior）的段
    div, mod = divmod(len(values), divisor)
    v1, c1 = [], []
    for i in range(div):
        v1.append(sum(values[divisor * i:divisor * i + divisor]) / divisor)
        c1.append(sum(counts[divisor * i:divisor * i + divisor]))
    if mod:
        v1.append(sum(values[divisor * i + divisor:]) / mod)
        c1.append(sum(counts[divisor * i + divisor:]))
    return (v1, c1)



'''#引用 dataloader类,计算各拓扑指标'''
def topo_result(path):
    Hyper_graph = dataloader(path)
    Hyper_graph.dataload()
    relevance_matrix = Hyper_graph.hyper_matrix.values
    num_node, num_edge = relevance_matrix.shape
    neibor_matrix = np.dot(relevance_matrix, (relevance_matrix).T)
    row, col = np.diag_indices_from(neibor_matrix)
    neibor_matrix[row, col] = 0

    diandu_list = {}  # 点度，没有重复算上不同超边中的相同相邻节点
    for i in range(num_node):
        diandu_list[i] = len(np.where(neibor_matrix[i] > 0)[0])
    # sort_diandu_list = sorted(diandu_list.items(), key=lambda item: item[1], reverse=True)  # 按点度值排倒序，得到列表，元素是（节点序号，节点度）

    dianqiangdu_list = {}  # 点强度，每个超边中的相邻节点都算上，有重复
    for i in range(num_node):
        dianqiangdu_list[i] = sum(neibor_matrix[i])
    # sort_dianqiangdu_list = sorted(dianqiangdu_list.items(), key=lambda item: item[1],
    #                                reverse=True)  # 按点度值排倒序，得到列表，元素是（节点序号，节点度）

    dianchaodu_list = {}  # 点超度，包含节点的超边数
    for i in range(num_node):
        dianchaodu_list[i] = sum(relevance_matrix[i])

    # sort_dianchaodu_list = sorted(dianchaodu_list.items(), key=lambda item: item[1],
    #                               reverse=True)  # 按点度值排倒序，得到列表，元素是（节点序号，节点度）

    HEdge_list = {}  # 超边基数，指超边包含的节点数量
    for i in range(num_edge):
        HEdge_list[i] = sum(relevance_matrix[:, i])

    # 点度
    values1, counts1 = np.unique(list(diandu_list.values()), return_counts=True)
    counts1 = counts1 / sum(counts1)

    # 点强度
    values2, counts2 = np.unique(list(dianqiangdu_list.values()), return_counts=True)
    counts2 = counts2 / sum(counts2)

    # 点超度
    values3, counts3 = np.unique(list(dianchaodu_list.values()), return_counts=True)
    counts3 = counts3 / sum(counts3)

    # 超边基数
    values4, counts4 = np.unique(list(HEdge_list.values()), return_counts=True)
    counts4 = counts4 / sum(counts4)
    return [(values1, counts1), (values2, counts2), (values3, counts3), (values4, counts4)]

def Fit_curve(datafile_path):
    '''#读取网络，绘制散点图，拟合曲线
    输入：数据文件夹路径
    输出：所有数据集在各个划分标准下的散点图以及合适拟合曲线
    保存：每一个数据集一个文件夹，每一个划分维度divisor=1/2/3/4一张图片'''
    datasets_name = os.listdir(datafile_path)
    for dataset in datasets_name[0:8]:  # 不同数据集
        # dataset = 'email-Enron.txt'
        path = datafile_path+"%s.txt" % dataset[:-4]
        four_datas = topo_result(path)
        for divisor in [1, 2, 3,4]:  # 同一数据集下的不同划分
            four_datas_divided = [Division(four_datas[i][0], four_datas[i][1], divisor) for i in range(len(four_datas))]
            # 一张图上展示
            plt.figure(figsize=(12, 8))
            for i in range(len(four_datas_divided)):
                data = four_datas_divided[i]
                plt.subplot(2, 2, i + 1)
                value, count = np.array(data[0]), np.array(data[1])

                # 依据RMSE选择拟合函数
                # functions = [linear_func, polynomial_func, exponential_func, power_func, logarithmic_func]
                # function_names = ['Linear Function', 'Polynomial Function', 'Exponential Function', 'Power Function',
                #                   'Logarithmic Function']
                # rmse_values = []
                # adjusted_r_squared_values = []

                # for func, func_name in zip(functions, function_names):
                #     if func == polynomial_func:
                #         # 对于多项式函数，需要提供初始系数猜测值
                #         p0 = np.ones(3)  # 这里假设是liu次多项式
                #         popt, pcov = curve_fit(func, value, count, p0=p0)
                #     else:
                #         popt, pcov = curve_fit(func, value, count)
                #     y_pred = func(value, *popt)
                #     rmse = calculate_rmse(count, y_pred)
                #     rmse_values.append(rmse)

                # # 找出具有最小RMSE的函数
                # min_rmse_index = np.argmin(rmse_values)
                # best_fit_func = functions[min_rmse_index]
                # best_fit_name = function_names[min_rmse_index]
                # value, count = np.array(data[0]), np.array(data[1])
                # if best_fit_func == polynomial_func:
                #     # 对于多项式函数，需要提供初始系数猜测值
                #     p0 = np.ones(3)  # 这里假设是liu次多项式
                #     best_fit_params, _ = curve_fit(best_fit_func, value, count,p0=p0)
                # else:
                #     best_fit_params, _ = curve_fit(best_fit_func, value, count)


                # 依据R函数选择函数
                functions = [linear_func, polynomial_func, exponential_func, power_func, logarithmic_func]
                function_names = ['Linear Function', 'Polynomial Function', 'Exponential Function', 'Power Function',
                                  'Logarithmic Function']

                adjusted_r_squared_values = []
                for func, func_name in zip(functions, function_names):
                    if func == polynomial_func:
                        # 对于多项式函数，需要提供初始系数猜测值
                        p0 = np.ones(3)  # 这里假设是三次多项式
                        popt, pcov = curve_fit(func, value, count, p0=p0)
                    else:
                        popt, pcov = curve_fit(func, value, count)
                    y_pred = func(value, *popt)
                    num_params = len(popt)
                    adjusted_r_squared = calculate_adjusted_r_squared(count, y_pred, num_params)
                    adjusted_r_squared_values.append(adjusted_r_squared)

                # 找出具有最大调整R平方的函数
                max_adjusted_r_squared_index = np.argmax(adjusted_r_squared_values)
                best_fit_func = functions[max_adjusted_r_squared_index]
                best_fit_name = function_names[max_adjusted_r_squared_index]
                if best_fit_func == polynomial_func:
                    # 对于多项式函数，需要提供初始系数猜测值
                    p0 = np.ones(3)  # 这里假设是三次多项式
                    best_fit_params, _ = curve_fit(best_fit_func, value, count,p0=p0)
                else:
                    best_fit_params, _ = curve_fit(best_fit_func, value, count)

                # 绘制四种拓扑散点图
                colors = ['b', 'r', 'g', 'purple']
                titles = ['Degree', 'Degree_Strength', 'Hyper_Degree', 'HEdge_size']
                plt.scatter(value, count, color=colors[i], marker='o')
                # 绘制拟合曲线，并标记拟合曲线的形式
                plt.plot(value, best_fit_func(value, *best_fit_params), color='red',
                         label='Best Fit Curve (' + best_fit_name + ')')
                # 设置双对数坐标
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(titles[i])
                plt.ylabel('Count')
                plt.legend()
                plt.title(titles[i] + ' Distribution')

                # 标记拟合曲线的函数表达式
                # fit_equation = best_fit_name + ': y = ' + ' + '.join(
                #     [f'{param:.4f} * x^{i}' for i, param in enumerate(best_fit_params)])
                # plt.text(0.5, 0.6, fit_equation, fontsize=12, transform=plt.gca().transAxes,
                #          bbox=dict(facecolor='white', alpha=0.5))

            # 自动调整图片，避免文字相互覆盖
            plt.tight_layout()
            # plt.show()
            if not os.path.exists('../result_topo2/%s' % dataset[:-4]):
                # 如果不存在，则新建一个名为A的文件夹
                os.makedirs('../result_topo2/%s' % dataset[:-4])
            else:
                pass
            plt.savefig('../result_topo2/%s/fit_%d_Distribution.jpg' % (dataset[:-4], divisor))


def topo_distribution(datafile_path):
    '''#读取网络，绘制散点图，拟合曲线
    输入：数据文件夹路径
    输出：所有数据集在各个划分标准下的散点图以及合适拟合曲线
    保存：每一个数据集一个文件夹，不划分d=1'''
    datasets_name = os.listdir(datafile_path)
    for dataset in datasets_name[0:8]:  # 不同数据集
        # dataset = 'email-Enron.txt'
        path = datafile_path + "%s.txt" % dataset[:-4]
        topo_datas = topo_result(path)
        # 一张图上展示
        plt.figure(figsize=(12, 8))
        for i in range(len(topo_datas)):
            data = topo_datas[i]
            plt.subplot(2, 2, i + 1)
            value, count = np.array(data[0]), np.array(data[1])

            # 依据RMSE选择拟合函数
            # functions = [linear_func, polynomial_func, exponential_func, power_func, logarithmic_func]
            # function_names = ['Linear Function', 'Polynomial Function', 'Exponential Function', 'Power Function',
            #                   'Logarithmic Function']
            # rmse_values = []
            # adjusted_r_squared_values = []

            # for func, func_name in zip(functions, function_names):
            #     if func == polynomial_func:
            #         # 对于多项式函数，需要提供初始系数猜测值
            #         p0 = np.ones(3)  # 这里假设是liu次多项式
            #         popt, pcov = curve_fit(func, value, count, p0=p0)
            #     else:
            #         popt, pcov = curve_fit(func, value, count)
            #     y_pred = func(value, *popt)
            #     rmse = calculate_rmse(count, y_pred)
            #     rmse_values.append(rmse)

            # # 找出具有最小RMSE的函数
            # min_rmse_index = np.argmin(rmse_values)
            # best_fit_func = functions[min_rmse_index]
            # best_fit_name = function_names[min_rmse_index]
            # value, count = np.array(data[0]), np.array(data[1])
            # if best_fit_func == polynomial_func:
            #     # 对于多项式函数，需要提供初始系数猜测值
            #     p0 = np.ones(3)  # 这里假设是liu次多项式
            #     best_fit_params, _ = curve_fit(best_fit_func, value, count,p0=p0)
            # else:
            #     best_fit_params, _ = curve_fit(best_fit_func, value, count)

            # 依据R函数选择函数
            functions = [linear_func, polynomial_func, exponential_func, power_func, logarithmic_func]
            function_names = ['Linear Function', 'Polynomial Function', 'Exponential Function', 'Power Function',
                              'Logarithmic Function']

            adjusted_r_squared_values = []
            for func, func_name in zip(functions, function_names):
                if func == polynomial_func:
                    # 对于多项式函数，需要提供初始系数猜测值
                    p0 = np.ones(3)  # 这里假设是三次多项式
                    popt, pcov = curve_fit(func, value, count, p0=p0)
                else:
                    popt, pcov = curve_fit(func, value, count)
                y_pred = func(value, *popt)
                num_params = len(popt)
                adjusted_r_squared = calculate_adjusted_r_squared(count, y_pred, num_params)
                adjusted_r_squared_values.append(adjusted_r_squared)

            # 找出具有最大调整R平方的函数
            max_adjusted_r_squared_index = np.argmax(adjusted_r_squared_values)
            best_fit_func = functions[max_adjusted_r_squared_index]
            best_fit_name = function_names[max_adjusted_r_squared_index]
            if best_fit_func == polynomial_func:
                # 对于多项式函数，需要提供初始系数猜测值
                p0 = np.ones(3)  # 这里假设是三次多项式
                best_fit_params, _ = curve_fit(best_fit_func, value, count, p0=p0)
            else:
                best_fit_params, _ = curve_fit(best_fit_func, value, count)

            # 绘制四种拓扑散点图
            colors = ['b', 'r', 'g', 'purple']
            titles = ['Degree', 'Degree_Strength', 'Hyper_Degree', 'HEdge_size']
            plt.scatter(value, count, color=colors[i], marker='o')
            # 绘制拟合曲线，并标记拟合曲线的形式
            plt.plot(value, best_fit_func(value, *best_fit_params), color='red',
                     label='Best Fit Curve (' + best_fit_name + ')')
            # 设置双对数坐标
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(titles[i])
            plt.ylabel('Count')
            plt.legend()
            plt.title(titles[i] + ' Distribution')

            # 标记拟合曲线的函数表达式
            # fit_equation = best_fit_name + ': y = ' + ' + '.join(
            #     [f'{param:.4f} * x^{i}' for i, param in enumerate(best_fit_params)])
            # plt.text(0.5, 0.6, fit_equation, fontsize=12, transform=plt.gca().transAxes,
            #          bbox=dict(facecolor='white', alpha=0.5))

        # 自动调整图片，避免文字相互覆盖
        plt.tight_layout()
        # plt.show()
        if not os.path.exists('../result_topo2/%s' % dataset[:-4]):
            # 如果不存在，则新建一个名为A的文件夹
            os.makedirs('../result_topo2/%s' % dataset[:-4])
        else:
            pass
        plt.savefig('../result_topo2/%s/Distribution.jpg' % (dataset[:-4]))

datafile_path = '../0-hypergraph_datasets/Ori_data/'
topo_distribution(datafile_path)
Fit_curve(datafile_path)