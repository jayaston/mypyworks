# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:07:35 2020

@author: Jay
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
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


