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


list1 = [['0023','00718','d']
               
         ]
shuju_df = tjfx.TjfxData().getdata('20190501','20200731',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
shuju_df['年月'] = shuju_df['QUOTA_DATE'].dt.strftime('%Y%m')
shuju_df['日'] = shuju_df['QUOTA_DATE'].dt.strftime('%d')
test = pd.pivot_table(shuju_df,index=['日'],columns = ['年月'],values='QUOTA_VALUE')
 

test.to_excel(r'C:\Users\XieJie\mypyworks\输出\2019-2020花都日供水量.xlsx')
