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


list1 = [['00','05960','h']
        
         ]
shuju_df = tjfx.TjfxData().getdata('20200101','20200224',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
#print(test)
test.info()
test.plot()
test.columns=['小时供水量']
test.resample('d').min().to_excel(r'C:\Users\XieJie\mypyworks\输出\2020夜间最小供水流量.xlsx')
test['小时'] = list(pd.Series(test.index).dt.strftime("%H"))
test.query("小时 )
test.sort_index()
test.to_excel(r'C:\Users\XieJie\mypyworks\输出\2015-2019售水.xlsx')
