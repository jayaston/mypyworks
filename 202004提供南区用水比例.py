# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import sys
import os
os.getcwd()
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx

#import re    
import datetime as dt


list1 = [['00','00409','m'],
         ['0902','00409','m'],
         ['00','05339','m'],
         ['04','05339','m'],
         ['05','05339','m'],
         ['00','05337','m'],
         ['04','05337','m'],
         ['05','05337','m'],
         ['00','05338','m'],
         ['04','05338','m'],
         ['05','05338','m'],
         ['00','05340','m'],
         ['04','05340','m'],
         ['05','05340','m'],
         ['00','05341','m'],
         ['04','05341','m'],
         ['05','05341','m'],
         ['1007','00718','m']         
         ]


shuju_df = tjfx.TjfxData().getdata('20170101','20191231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
#print(test)
test.info()
#test.plot()
#plt.axhline(y=583357,ls="--")#添加水平直线

##按月份删选
#test['mon'] = test.index.strftime('%m')
#test = test[test.mon<='10']
#test.drop([("mon","")],axis=1,inplace=True)
#按照指定的顺序对列排序
#test = test[['西村水厂','石门水厂','江村水厂','新塘水厂','西洲水厂','南洲水厂']]
test = test.resample("Y").sum()

test.to_excel(r'C:\Users\XieJie\mypyworks\输出\2017-2019南区用水比例.xlsx')