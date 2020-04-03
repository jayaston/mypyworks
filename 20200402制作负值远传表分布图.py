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
data = pd.read_excel(
        r'e:\专题工作（重要）\公司级大数据系统资料\战略与大数据分析系统项目\远传表数据治理\相邻日期行度相减小于-1的情况.xlsx',
        index_col=0
        )

data



test = pd.pivot_table(data,index=['数据时间'],columns=['OBJID'],values=['行度值'],aggfunc=[len],fill_value=0)





list1=data['OBJID'].unique().tolist()

import random
slicelist = random.sample(list(test.columns),10)
testslice = test[slicelist]

testarray = testslice.values.T
plt.figure(figsize=(12,12))
plt.imshow(testarray)
#plt.colorbar()
plt.show()    
