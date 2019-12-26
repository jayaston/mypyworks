# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 16:20:31 2019

@author: XieJie
"""

#时间序列的异常值判断
import numpy as np
def moving_average(a, n=3): 
    ret = np.cumsum(np.pad(a, n, 'edge'), dtype=float) 
    ret[n:] = ret[n:] - ret[:-n] 
    return ret[n:-n] / n

am = moving_average(a, n=15) #离群点少的话
np.argwhere(np.abs(a-am)>3 * np.std(a-am))

#聚类异常值判断
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as mp


def get_data_zs(inputfile):
    data = pd.read_excel(inputfile, index_col='Id', encoding='gb18030')
    data_zs = 1.0 * (data - data.mean()) / data.std()
    return data, data_zs


def model_data_zs(data, k, b):
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=b)
    model.fit(data_zs)

    # 标准化数据及其类别
    r = pd.concat(
        [data_zs, pd.Series(model.labels_, index=data.index)], axis=1)
    # print(r.head())
    # 每个样本对应的类别
    r.columns = list(data.columns) + [u'聚类类别']  # 重命名表头
    return model, r, k


def make_norm(model, k):
    norm = []
    for i in range(k):
        norm_tmp = r[['R', 'F', 'M']][
            r[u'聚类类别'] == i] - model.cluster_centers_[i]
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)  # 求出绝对距离
        norm.append(norm_tmp / norm_tmp.median())  # 求相对距离并添加
    norm = pd.concat(norm)
    return norm


def draw_discrete_point(threshold):
    mp.rcParams['font.sans-serif'] = ['SimHei']
    mp.rcParams['axes.unicode_minus'] = False
    norm[norm <= threshold].plot(style='go')  # 正常点

    discrete_points = norm[norm > threshold]  # 离散点阈值
    discrete_points.plot(style='rs')
    # print(discrete_points)

    for i in range(len(discrete_points)):  # 离群点做标记
        id = discrete_points.index[i]
        n = discrete_points.iloc[i]
        mp.annotate('(%s,%0.2f)' % (id, n), xy=(id, n), xytext=(id, n))
    mp.xlabel(r'编号')
    mp.ylabel(r'相对距离')
    mp.show()

if __name__ == '__main__':
    inputfile = 'data/consumption_data.xls'
    threshold = 2 # 离散点阈值
    k = 3 # 聚类类别
    b = 500 # 聚类最大循环次数
    data, data_zs = get_data_zs(inputfile)
    model, r, k = model_data_zs(data, k, b)
    norm = make_norm(model, k)
    draw_discrete_point(threshold)
    print('All Done')