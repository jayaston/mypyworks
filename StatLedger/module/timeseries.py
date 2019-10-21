# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 07:46:30 2019

@author: Jay
"""
import pandas as pd 
import numpy as np 
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import datetime as dt
import tjfxdata as tjfx

list = [['00','00718','d']]
shuju_df = tjfx.TjfxData().getdata('20160101','20191015',list)
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
df = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
df.columns 
train = df[df.index <= dt.datetime.strptime('2018-12-31','%Y-%m-%d')]
test = df[df.index > dt.datetime.strptime('2018-12-31','%Y-%m-%d')]
train['公司供水总量'].plot( title= '公司供水总量变化', fontsize=15,kind="bar")
test['公司供水总量'].plot( fontsize=15)

