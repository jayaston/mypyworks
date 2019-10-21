# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:14:19 2019

@author: Jay
"""

import pandas as pd
import numpy as np

data_df = pd.read_csv('./自来水数据/查询指标.csv',header=None,
                      dtype={'QUOTA_CODE' : str,'DEPT_CODE':str},                      
                      usecols = ['QUOTA_CODE','QUOTA_DATE','QUOTA_VALUE','DEPT_CODE','RECORD_TYPE'],
                      names=['QUOTA_CODE','MON','QUOTA_DATE','QUOTA_VALUE','忽略1','DEPT_CODE','忽略2','忽略3','RECORD_TYPE'])
data_df.info()
data_df.head()
#data_df['DATE'] = pd.to_datetime(data_df['QUOTA_DATE'],format = '%d/%m/%Y %H:%M:%S' ,errors = 'coerce')



dept_df = pd.read_csv('./自来水数据/查询指标_部门编码.csv',header=None,
                      names = ['DEPT_CODE','DEPT_NAME'],
                      dtype={'DEPT_CODE':str,'DEPT_NAME':str})

dept_df.drop_duplicates('DEPT_CODE',inplace=True)


quota_df = pd.read_csv('./自来水数据/查询指标_指标编码.csv',header=None,
                       usecols=[0,1,11],names=['QUOTA_CODE','QUOTA_NAME','effect'],
                       dtype={'QUOTA_CODE':object,'QUOTA_NAME':object} )


quota_df.drop_duplicates('QUOTA_CODE',inplace=True)
quota_df = quota_df.query("effect=='Y'")[['QUOTA_CODE','QUOTA_NAME']]
df = pd.merge(data_df,dept_df,on = 'DEPT_CODE')

#test =df[~(df['QUOTA_CODE'].isin(list(quota_df['QUOTA_CODE'])))]
#test.tail()
df = pd.merge(df,quota_df,on='QUOTA_CODE')
df.info()
df.to_csv('./自来水数据/整理后指标.csv')