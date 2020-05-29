# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:32:14 2020

@author: XieJie
"""

localPath = r'c:/Users/XieJie/mypyworks/StatLedger/'
prevdays = 30
import sys
sys.path.append(localPath+r'module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx
#import re    
import datetime as dt
enddate = (dt.datetime.now()-dt.timedelta(days=1)).strftime('%Y%m%d')
startdate = (dt.datetime.now()-dt.timedelta(days=prevdays)).strftime('%Y%m%d')

shuju_df = tjfx.TjfxData().getdata(startdate,enddate)
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce')#可以带.fillna()
shuju_df = shuju_df[np.isfinite(shuju_df.QUOTA_VALUE)]
shuju_df = shuju_df.query("QUOTA_VALUE != 0")

#selectquota = pd.read_excel(localPath+r'数据表\数据集_水厂监控看板.xlsx',
#                            dtype={'QUOTA_CODE':str},
#                            usecols=['QUOTA_CODE'])
#selectquotalist = list(set(selectquota['QUOTA_CODE']))
#shuju_df = shuju_df[shuju_df['QUOTA_CODE'].isin(selectquotalist)]

shuju_pivot_m = shuju_df.query("RECORD_TYPE=='m'")
shuju_pivot_m = pd.pivot_table(shuju_pivot_m,index = ['QUOTA_DATE'],
                      columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE',fill_value=0)
shuju_pivot_m = shuju_pivot_m.reset_index()
shuju_pivot_m.columns = list(map(lambda X : '_'.join(X),list(shuju_pivot_m.columns)))

shuju_pivot_d = shuju_df.query("RECORD_TYPE=='d'")
shuju_pivot_d = pd.pivot_table(shuju_pivot_d,index = ['QUOTA_DATE'],
                      columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE',fill_value=0)
shuju_pivot_d = shuju_pivot_d.reset_index()
shuju_pivot_d.columns = list(map(lambda X : '_'.join(X),list(shuju_pivot_d.columns)))

shuju_pivot_h = shuju_df.query("RECORD_TYPE=='h'")
shuju_pivot_h = pd.pivot_table(shuju_pivot_h,index = ['QUOTA_DATE'],
                      columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE',fill_value=0)
shuju_pivot_h = shuju_pivot_h.reset_index()
shuju_pivot_h.columns = list(map(lambda X : '_'.join(X),list(shuju_pivot_h.columns)))

riqibiao = pd.DataFrame({'日期时间':pd.date_range(startdate,(dt.datetime.now()-dt.timedelta(days=1)).strftime('%Y%m%d 23:00:00'),freq='h')})
riqibiao = riqibiao.assign(年度 = riqibiao['日期时间'].dt.strftime("%Y"),
                           年度季度 = riqibiao['日期时间'].dt.strftime("%Y") + "Q"+ ((riqibiao['日期时间'].dt.strftime("%m").astype("int32")-1)//3 + 1).astype(str),
                           年度月份 = riqibiao['日期时间'].dt.strftime("%Y%m"),
                           日期 = riqibiao['日期时间'].dt.strftime("%Y-%m-%d"))