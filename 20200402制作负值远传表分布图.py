# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:51:22 2020

@author: XieJie
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
data = pd.read_excel(
        r'c:\Users\Jay\mypyworks\自来水数据\相邻日期行度相减小于-1的情况.xlsx',
        index_col=0
        )

data['数据时间']=data['数据时间'].dt.strftime("%y/%m/%d")
data['OBJID'] = data['OBJID'].astype('str')

test = pd.pivot_table(data,index=['OBJID'],columns=['数据时间'],values=['水量'],fill_value=0)

test[test<0] = np.nan

test.to_excel('./mypyworks/数据导出/相邻日期行度相减小于-1矩阵.xlsx')










list1=data['OBJID'].unique().tolist()

import random
slicelist = random.sample(list(test.columns),10)
testslice = test[slicelist]

testarray = testslice.values.T
plt.figure(figsize=(12,12))
plt.imshow(testarray)
#plt.colorbar()
plt.show()    
