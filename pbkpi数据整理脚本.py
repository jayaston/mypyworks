# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:03:57 2021

@author: xiejie
"""

import pandas as pd
import numpy as np
import datetime as dt
#公司
data = pd.read_excel(r'd:\专题工作（重要）\产销差工作\产销差数据与资料积累\数据汇总(未加2106调整的中区耗水量用excel打开).xlsx',
                      sheet_name='DM_计划完成情况',usecols="A:M,W:Y",skiprows=14,nrows=24,
                      names=['月份','月供水总量计划','月供水总量','累计供水总量计划','累计供水总量',
                             '月售水量计划','月售水量','累计售水量计划','累计售水量',
                             '月产销差率计划','月产销差率','累计产销差率计划','累计产销差率',
                             '月终端漏损率','累计终端漏损率计划','累计终端漏损率'])
data.eval('''
            月产销差水量 = 月供水总量-月售水量
            累计产销差水量  = 累计供水总量 - 累计售水量
            月漏损水量 = 月供水总量*月终端漏损率
            累计漏损水量 = 累计供水总量*累计终端漏损率
            月免费水量  = 月产销差水量 - 月漏损水量
            累计免费水量  = 累计产销差水量 - 累计漏损水量
            月供水总量为计划 = 月供水总量/月供水总量计划-1
            月售水量为计划  = 月售水量/月售水量计划-1
            累计售水量为计划  = 累计售水量/累计售水量计划-1
            累计供水总量为计划 = 累计供水总量/累计供水总量计划-1
            累计产销差率为计划  = 累计产销差率 - 累计产销差率计划
            累计终端漏损率为计划  = 累计终端漏损率 - 累计终端漏损率计划
            ''',inplace=True)



#添加同比数
data1 = data[['月供水总量','累计供水总量','月售水量','累计售水量','月产销差水量','累计产销差水量',
              '月免费水量','累计免费水量','月漏损水量','累计漏损水量']].pct_change(periods=12)
data1.columns=['月供水总量同比','累计供水总量同比','月售水量同比','累计售水量同比','月产销差水量同比','累计产销差水量同比',
              '月免费水量同比','累计免费水量同比','月漏损水量同比','累计漏损水量同比']

data2 = data[['月产销差率','累计产销差率','月终端漏损率','累计终端漏损率']].diff(periods=12)
data2.columns=['月产销差率同比','累计产销差率同比','月终端漏损率同比','累计终端漏损率同比']
#添加同期数
data3 = data[['月供水总量','累计供水总量','月售水量','累计售水量','月产销差水量','累计产销差水量','月产销差率','累计产销差率',
              '月免费水量','累计免费水量','月终端漏损率','累计终端漏损率','月漏损水量','累计漏损水量']].shift(periods=12, axis=0)
data3.columns=['月供水总量同期','累计供水总量同期','月售水量同期','累计售水量同期','月产销差水量同期','累计产销差水量同期','月产销差率同期','累计产销差率同期',
              '月免费水量同期','累计免费水量同期','月终端漏损率同期','累计终端漏损率同期','月漏损水量同期','累计漏损水量同期']
#连接
data = pd.concat([data,data1,data2,data3],axis=1)

data = data[12:]

kpiname = ['月供水总量为计划','累计供水总量为计划','月售水量为计划','累计售水量为计划','累计产销差率为计划','累计终端漏损率为计划',
           '月供水总量同比','累计供水总量同比','月售水量同比','累计售水量同比','月产销差水量同比',
           '累计产销差水量同比','月免费水量同比','累计免费水量同比','月漏损水量同比','累计漏损水量同比','月产销差率同比','累计产销差率同比','月终端漏损率同比','累计终端漏损率同比']

data4=data[kpiname].apply(
    lambda x: pd.cut(x,bins=[-np.inf,-0.07,0,0.07 ,np.inf],labels=[1, 2, 3, 4]),axis=0)

data4.columns=list(map(lambda x: x+'index',kpiname))
data = pd.concat([data,data4],axis=1)
n=12#指标数量
#data['id']=list(range(1,n+1))*(len(data)//n)+list(range(1,len(data)%n+1))
data['id']=1

data_all = data

#中区
data = pd.read_excel(r'd:\专题工作（重要）\产销差工作\产销差数据与资料积累\数据汇总(未加2106调整的中区耗水量用excel打开).xlsx',
                      sheet_name='DM_计划完成情况',usecols="A,AB:AM,W:Y",skiprows=14,nrows=24,
                      names=['月份','月供水总量计划','月供水总量','累计供水总量计划','累计供水总量',
                             '月售水量计划','月售水量','累计售水量计划','累计售水量',
                             '月产销差率计划','月产销差率','累计产销差率计划','累计产销差率',
                             '月终端漏损率','累计终端漏损率计划','累计终端漏损率'])
data.eval('''
            月产销差水量 = 月供水总量-月售水量
            累计产销差水量  = 累计供水总量 - 累计售水量
            月漏损水量 = 月供水总量*月终端漏损率
            累计漏损水量 = 累计供水总量*累计终端漏损率
            月免费水量  = 月产销差水量 - 月漏损水量
            累计免费水量  = 累计产销差水量 - 累计漏损水量
            月供水总量为计划 = 月供水总量/月供水总量计划-1
            月售水量为计划  = 月售水量/月售水量计划-1
            累计售水量为计划  = 累计售水量/累计售水量计划-1
            累计供水总量为计划 = 累计供水总量/累计供水总量计划-1
            累计产销差率为计划  = 累计产销差率 - 累计产销差率计划
            累计终端漏损率为计划  = 累计终端漏损率 - 累计终端漏损率计划
            ''',inplace=True)



#添加同比数
data1 = data[['月供水总量','累计供水总量','月售水量','累计售水量','月产销差水量','累计产销差水量',
              '月免费水量','累计免费水量','月漏损水量','累计漏损水量']].pct_change(periods=12)
data1.columns=['月供水总量同比','累计供水总量同比','月售水量同比','累计售水量同比','月产销差水量同比','累计产销差水量同比',
              '月免费水量同比','累计免费水量同比','月漏损水量同比','累计漏损水量同比']

data2 = data[['月产销差率','累计产销差率','月终端漏损率','累计终端漏损率']].diff(periods=12)
data2.columns=['月产销差率同比','累计产销差率同比','月终端漏损率同比','累计终端漏损率同比']
#添加同期数
data3 = data[['月供水总量','累计供水总量','月售水量','累计售水量','月产销差水量','累计产销差水量','月产销差率','累计产销差率',
              '月免费水量','累计免费水量','月终端漏损率','累计终端漏损率','月漏损水量','累计漏损水量']].shift(periods=12, axis=0)
data3.columns=['月供水总量同期','累计供水总量同期','月售水量同期','累计售水量同期','月产销差水量同期','累计产销差水量同期','月产销差率同期','累计产销差率同期',
              '月免费水量同期','累计免费水量同期','月终端漏损率同期','累计终端漏损率同期','月漏损水量同期','累计漏损水量同期']
#连接
data = pd.concat([data,data1,data2,data3],axis=1)

data = data[12:]

kpiname = ['月供水总量为计划','累计供水总量为计划','月售水量为计划','累计售水量为计划','累计产销差率为计划','累计终端漏损率为计划',
           '月供水总量同比','累计供水总量同比','月售水量同比','累计售水量同比','月产销差水量同比',
           '累计产销差水量同比','月免费水量同比','累计免费水量同比','月漏损水量同比','累计漏损水量同比','月产销差率同比','累计产销差率同比','月终端漏损率同比','累计终端漏损率同比']

data4=data[kpiname].apply(
    lambda x: pd.cut(x,bins=[-np.inf,-0.07,0,0.07 ,np.inf],labels=[1, 2, 3, 4]),axis=0)

data4.columns=list(map(lambda x: x+'index',kpiname))
data = pd.concat([data,data4],axis=1)
n=12#指标数量
#data['id']=list(range(1,n+1))*(len(data)//n)+list(range(1,len(data)%n+1))
data['id']=1

data_all = pd.concat([data_all,data],axis=0)