# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:07:35 2020

@author: Jay
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
import sxtwl



#计算农历日期
lunar = sxtwl.Lunar()
solar_day = lunar.getDayByLunar(2020,2,1)
dt.date(solar_day.y, solar_day.m, solar_day.d)

dt.datetime(2019,6,9)-dt.datetime(2019,2,3)+dt.timedelta(1)
dt.datetime(2020,6,27)-dt.datetime(2020,1,23)+dt.timedelta(1)


#202002月内容
#夜间最小流量和最大流量数据获取计算
list1 = [['00','05960','h']        
         ]
shuju_df = tjfx.TjfxData().getdata('20190101','20200628',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
#print(test)
test.info()
test.plot()
test.columns=['小时供水量']
test.sort_index()
test.resample('d').min().to_excel(
        r'C:\Users\XieJie\mypyworks\输出\2019-2020夜间最小供水流量.xlsx')
test['小时'] = list(pd.Series(test.index).dt.strftime("%H"))
test.query("小时 in ['07','08']").resample('d')['小时供水量'].sum(
        ).to_excel(r'C:\Users\XieJie\mypyworks\输出\2019-2020年7-9时供水流量.xlsx')



#远传表的行业售水情况
df1 = pd.read_excel(r'C:\Users\Jay\Desktop\远程办公\新冠疫情影响供水行业数据\数据.xlsx',sheet_name=3)
df1.query("用途 == '计费'")['性质'].drop_duplicates()

df1.query("用途 == '计费'").groupby(['性质'])['合计'].sum().sort_values(ascending=False)





df2 = df1.query("用途 == '计费' & 性质 in ['居民生活','经营服务用水','学校用水','工业用水']")
df2.columns
df2 = df2.groupby(['性质'])[['2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14',
           '2020-01-15','2020-01-16', '2020-01-17', '2020-01-18', '2020-01-19', 
           '2020-01-20', '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24',
           '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29',
           '2020-01-30','2020-01-31', '2020-02-01', '2020-02-02', '2020-02-03' 
           ]].sum().T

plt.figure(figsize=(12,8))               
plt.subplot(221)
plt.bar(x=pd.to_datetime(df2.index,format='%Y-%m-%d'),height=df2['学校用水'])
plt.title('学校用水')
plt.grid(which='both', linestyle='--')
plt.subplot(222)
plt.bar(x=pd.to_datetime(df2.index,format='%Y-%m-%d'),height=df2['居民生活'])
plt.title('居民生活')
plt.grid(which='both', linestyle='--')
plt.subplot(223)
plt.bar(x=pd.to_datetime(df2.index,format='%Y-%m-%d'),height=df2['工业用水'])
plt.grid(which='both', linestyle='--')
plt.title('工业用水')
plt.subplot(224)
plt.bar(x=pd.to_datetime(df2.index,format='%Y-%m-%d'),height=df2['经营服务用水'])
plt.grid(which='both', linestyle='--')
plt.title('经营服务用水')
plt.show()

df2 = df1.query("用途 == '计费' & 性质 in ['居民生活','经营服务用水','学校用水','工业用水']")





#6月份更新内容
#月度行业售水情况
list1 = [['00','12316','m'],#公司学校
         ['00','04861','m'],#公司学校用水
         ['00','05337','m'],#公司居民
         ['00','05339','m'],#公司一般工业
         ['00','05340','m']#公司经营服务（19年11前）         
         ]
shuju_df = tjfx.TjfxData().getdata('20191001','20200627',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

df3 = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=['QUOTA_NAME'])#提取的行业数据横置

df3.columns = ['一般工业','学校水量','学校用水水量','居民用水','经营服务业']#规范名称
df3['学校'] = df3['学校水量']+df3['学校用水水量']#汇总学校水量
df3 = df3[['一般工业','学校','居民用水','经营服务业']]#选择需要字段
df3.info()

plt.figure(figsize=(12,8))               
plt.subplot(221)
plt.bar(x=df3.index.strftime('%Y-%m'),height=df3['学校'])
plt.title('学校用水')
plt.grid(which='both', linestyle='--')
plt.subplot(222)
plt.bar(x=df3.index.strftime('%Y-%m'),height=df3['居民用水'])
plt.title('居民生活')
plt.grid(which='both', linestyle='--')
plt.subplot(223)
plt.bar(x=df3.index.strftime('%Y-%m'),height=df3['一般工业'])
plt.grid(which='both', linestyle='--')
plt.title('工业用水')
plt.subplot(224)
plt.bar(x=df3.index.strftime('%Y-%m'),height=df3['经营服务业'])
plt.grid(which='both', linestyle='--')
plt.title('经营服务用水')
plt.show()

#累计远传大表20册水量统计
list1 = [['00','30953','m']#20册远传大表水量                
         ]
shuju_df = tjfx.TjfxData().getdata('20190101','20200627',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

df3 = pd.pivot_table(shuju_df,index=['QUOTA_DATE'],columns=['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')#

df3.to_excel(r'C:\Users\XieJie\mypyworks\输出\E20远传大表20册水量.xlsx')