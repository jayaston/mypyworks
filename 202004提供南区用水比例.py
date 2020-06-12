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
         #['00','00409','m'],#公司售水量
         ['0902','00409','m'],#南区售水量
         #['00','05339','m'],#公司一般工业
         ['04','05339','m'],#海珠片工业
         ['05','05339','m'],#芳村片工业
         #['00','05337','m'],#公司居民
         ['04','05337','m'],#海珠居民         
         ['05','05337','m'],#芳村居民
         #['00','05338','m'],#公司行政
         ['04','05338','m'],#海珠行政
         ['05','05338','m'],#芳村行政
         
         ['04','12316','m'],#海珠学校
         ['04','04861','m'],#海珠学校用水
         ['04','12320','m'],#海珠消防
         ['04','04870','m'],#海珠行政事业
         ['04','05755','m'],#海珠行政
         ['04','12342','m'],#海珠趸售行政
         ['05','12316','m'],#芳村学校
         ['05','04861','m'],#芳村学校用水
         ['05','12320','m'],#芳村消防
         ['05','04870','m'],#芳村行政事业
         ['05','05755','m'],#芳村行政
         ['05','12342','m'],#芳村趸售行政
         
         ['04','04855','m'],#海珠环卫
         ['04','12318','m'],#海珠绿化
         ['05','04855','m'],#芳村环卫
         ['05','12318','m'],#芳村绿化
         
         ['04','04862','m'],#海珠公园
         ['05','04862','m'],#芳村公园
         
         ['04','04859','m'],#海珠医疗
         ['05','04859','m'],#芳村医疗
         
         #['00','05340','m'],#公司经营服务（19年11前）
         ['04','05340','m'],#海珠经营服务（19年11前）
         ['05','05340','m'],#芳村经营服务（19年11前）
         #['00','05341','m'],#公司特种（19年11前）
         ['04','05341','m'],#海珠特种（19年11前）
         ['05','05341','m'],#芳村特种（19年11前）
         ['1007','00718','m']#南洲供水总量         
         ]


shuju_df = tjfx.TjfxData().getdata('20160101','20191231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE',fill_value=0)
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