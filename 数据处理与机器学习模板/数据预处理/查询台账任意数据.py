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


list1 = [         
         ['00','31195','m'],
         ['1001','31195','m'],
         ['1002','31195','m'],
         ['1003','31195','m'],
         ['1004','31195','m'],
         ['1005','31195','m'],
         ['1007','31195','m'],
         ['1016','31195','m'],
         ['1009','31195','m']
         ]
shuju_df = tjfx.TjfxData().getdata('20190101','20191231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
list1 = [x+'_'+y for x,y in zip(test.columns.get_level_values(0).values , test.columns.get_level_values(1).values)]  
test.columns = list1

test = test.resample('Y').sum()

test.to_excel(r'C:\Users\XieJie\mypyworks\输出\2019年各厂各月取供比.xlsx')

#求最大日
list2 = [['1009','04281','d']
         
         ]
shuju_df = tjfx.TjfxData().getdata('20140101','20191231',list2)
shuju_df.info()
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

shuju_df2 = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=[ 'QUOTA_NAME'],values= 'QUOTA_VALUE',
                          aggfunc=np.sum)
shuju_df2.reset_index(inplace=True)
shuju_df2.info()
#按年最大值
get_max = lambda x: x[x['入厂取水量']==x['入厂取水量'].max()][['QUOTA_DATE','入厂取水量']]
                        
# python就是灵活啊。
get_max.__name__ = "maxday"
test = shuju_df2.groupby([pd.Grouper(key='QUOTA_DATE',freq='Y')]).apply(get_max)


