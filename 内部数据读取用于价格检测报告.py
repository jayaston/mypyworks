# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""
import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx
#import re    



list1 = [['00','00409','m'],
         ['00','00718','m'],
         ['00','11930','m'],
         ['00','05337','m'],
         ['00','05338','m'],
         ['00','05340','m'],
         ['00','05339','m'],
         ['00','05341','m'],
         ['00','04662','m'],#染织水量
         ['00','04664','m'],#化工水量
         ['00','04666','m'],#冶炼水量
         ['00','01168','m'],#漏损水量
         ['00','01169','m'],#漏损率
         ['0900','00409','m'],#东区售水量
         ['0901','00409','m'],#中区售水量
         ['0902','00409','m'],#南区售水量
         ['0903','00409','m']]#北区售水量
shuju_df = tjfx.TjfxData().getdata('20180501','20181231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)
shuju_df.to_excel('E:\\pyworks\\价格监测内部数据提取.xls')

neibushuju = pd.read_excel('E:\\水价2018\\修改建议二（一）基本情况部分数据\\内部数据.xlsx')
neibushuju.info()
neibushuju['年份'] = neibushuju['月份'].map(lambda x: x[:4])
data = neibushuju.groupby('年份')['售水总量（m3）'].sum().reset_index()
data.info()
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.ylim(1000000000,1250000000)
plt.xlabel('年份')
plt.ylabel('水量(10亿立方米)')
plt.title("广州市2008-2018售水情况")
plt.bar(data['年份'],data['售水总量（m3）'])
plt.show()




