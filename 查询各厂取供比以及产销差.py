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


list1 = [['00','11930','m'],         
         ['00','00718','m'],
         ['1009','04281','m'],
         ['1009','31199','m'],
         ['1001','04281','m'],
         ['1001','00718','m'],
         ['1002','04281','m'],
         ['1002','00718','m'],
         ['1003','04281','m'],
         ['1003','00718','m'],
         ['1004','04281','m'],
         ['1004','00718','m'],
         ['1005','04281','m'],
         ['1005','00718','m'],
         ['1007','04281','m'],
         ['1007','00718','m'],
         ['1016','04281','m'],
         ['1016','00718','m'],
         ['00','00409','m']
         ]
shuju_df = tjfx.TjfxData().getdata('20170101','20191231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
#print(test)
test.resample('Y').sum().to_excel(r'C:\Users\XieJie\mypyworks\输出\20200316吴部需要各厂取供比.xlsx')

test.info()

