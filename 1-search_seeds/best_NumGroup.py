import os
import matplotlib
# 可视化展示
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 在数据集循环外层创建结果容器
def performanceANDtime(Num_Group_list):
    results = []
    times = []
    datasets_name = os.listdir('../0-hypergraph_datasets/Ori_data/')
    for dataset in datasets_name[:]:
        dataset_results = []
        dataset_times = []
        for Num_Group in Num_Group_list:
            # 读取数据文件

            file_path = f'D:/MyOneDrive/OneDrive/桌面/01-Experiments/1-search_seeds/robustness_result/{Num_Group}{dataset[:-4]}.xlsx'
            df = pd.read_excel(file_path,
                               sheet_name="MHDP_influence",
                               index_col=0,
                               usecols=[0, 1],
                               skipfooter=1)
            df_time = pd.read_excel(file_path,
                               sheet_name="Per_Cost_Time",
                               index_col=0,
                               usecols=[0, 1],
                               skipfooter=1)
            # 计算当前组合的平均值
            avg_value = df.iloc[:, 0].mean()  # 假设数据在第一列
            dataset_results.append(avg_value)

            avg_time = df_time.iloc[-1, 0] / (len(df_time) - 1)
            dataset_times.append(avg_time)
            # 打印进度
            print(f"Dataset: {dataset} | NG Value: {Num_Group} | Average: {avg_value:.2f} | Average_time: {avg_time:.2f}")

            # 保存当前数据集的结果
        results.append({
            'dataset': dataset[:-4],  # 去除.txt后缀
            'averages': dataset_results
        })
        times.append({
            'dataset': dataset[:-4],  # 去除.txt后缀
            'averages_time': dataset_times
        })

    # 转换为DataFrame便于分析
    param_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    result_df = pd.DataFrame({
        res['dataset']: res['averages'] for res in results
    }, index=param_values).T
    time_df = pd.DataFrame({
            res['dataset']: res['averages_time'] for res in times
        }, index=param_values).T
    # ====== 新增代码开始 ======
    # # 计算增长率（100相比20）
    # result_df['Growth_Rate_%'] = ((result_df[100] - result_df[5]) / result_df[5]) * 100
    #
    # # 格式化显示为百分比（保留两位小数）
    # growth_report = result_df['Growth_Rate_%'].round(2).astype(str) + '%'
    #
    # # 打印报告
    # print("\nGrowth Rate Report (Num_Group 100 vs 20):")
    # print(growth_report)
    # ====== 新增代码结束 ======
    # 保存结果到Excel
    result_df.to_excel('parameter_performance.xlsx')
    time_df.to_excel('parameter_time.xlsx')

def plot_NG_scatter(result_df, time_df, Num_Group_list):
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 16
    plt.rcParams['mathtext.fontset'] = 'cm'

    # 创建画布和子图布局
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25,
                          top=0.88, bottom=0.08,
                          left=0.07, right=0.93)

    # 创建颜色映射（优化版冷-暖渐变）
    cmap = LinearSegmentedColormap.from_list("cool_warm", [
        '#313695',  # 深蓝
        '#4575b4',  # 蓝
        '#74add1',  # 浅蓝
        '#abd9e9',  # 天蓝
        '#e0f3f8',  # 淡蓝
        '#fee090',  # 黄
        '#fdae61',  # 橙
        '#f46d43',  # 红橙
        '#d73027',  # 红
        '#a50026'  # 深红
    ])

    # 创建标准化映射
    norm = plt.Normalize(vmin=5, vmax=100)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # 绘制子图
    datasets = result_df.index.tolist()[:8]
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        values = result_df.loc[dataset].values
        times = time_df.loc[dataset].values

        # 绘制散点（使用统一颜色映射）
        sc = ax.scatter(times, values, c=Num_Group_list,
                        cmap=cmap, norm=norm,
                        s=70, edgecolor='w',
                        linewidth=0.5, alpha=0.8)

        # 子图标签设置
        ax.set_title(dataset, fontsize=16, pad=10)
        ax.set_xlabel('Time Cost', fontsize=16)
        ax.set_ylabel('Propagation Scale', fontsize=16)
        ax.grid(True, alpha=0.3, linestyle=':')

    # 添加全局colorbar（精确对齐子图宽度）
    cax = fig.add_axes([
        0.07,  # 左边界与子图对齐
        0.95,  # 位于顶部下方
        0.86,  # 宽度与4列子图总宽度一致
        0.02  # 高度
    ])
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('$NG$', labelpad=5, fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # 优化刻度标签显示
    tick_positions = Num_Group_list
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f'{x:.1f}' for x in tick_positions])

    plt.savefig('NG_scatter_analysis.png', dpi=600, bbox_inches='tight')
    plt.show()

Num_Group_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
performanceANDtime(Num_Group_list)
result_df = pd.read_excel('parameter_performance.xlsx', index_col=0)
time_df = pd.read_excel('parameter_time.xlsx', index_col=0)
plot_NG_scatter(result_df, time_df, Num_Group_list)
