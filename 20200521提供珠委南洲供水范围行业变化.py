# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:53:40 2020

@author: XieJie
"""
import pandas as pd
df = pd.read_excel(
    r'e:\\专题工作（重要）\取水资料\各水源取水档案资料存档\北江资料\20200515西海取水点项目取用水监督检查\工业用水\标识行业\2019年海珠片月均水量超5000客户清单.xls',
    index_col = 0)
df_gongye = df.query("五大行业 == '工业' & 水量 != 0")
test = df_gongye.groupby(['客户编号'])[['客户编号','水量']].mean()

kehu = test[test['水量']>=50000]['客户编号'].values


df_gongye[df_gongye['客户编号'].isin(kehu)]['水量'].sum()/10000/3784.90879227962


