# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:19:57 2019

@author: Jay
"""
localPath = r"c:/Users/XieJie/mypyworks/StatLedger"
import sys
sys.path.append(localPath+r'/module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx
#import re    
import datetime as dt
enddate = (dt.datetime.now()-dt.timedelta(days=1)).strftime('%Y%m%d')
startdate = (dt.datetime.now()-dt.timedelta(days=731)).strftime('%Y%m%d')

shuju_df = tjfx.TjfxData().getdata(startdate,enddate)
shuju_df = shuju_df.query("RECORD_TYPE!='h'")
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)



selectquota = pd.read_excel(localPath+r"数据表\水厂看板指标.xlsx",
                            dtype={'QUOTA_CODE':object,'QUOTA_NAME':object},
                            usecols=['QUOTA_CODE'])

shuju_df = pd.merge(shuju_df,selectquota,on="QUOTA_CODE")
shuju_df = pd.pivot_table(shuju_df,index = ['QUOTA_DATE','GROUP_NAME','RECORD_TYPE'],
                      columns = ['QUOTA_NAME'],values='QUOTA_VALUE',fill_value=0)
shuju_df = shuju_df.reset_index()


riqibiao = pd.DataFrame({'日期':pd.date_range(startdate,enddate)})
riqibiao = riqibiao.assign(年度 = riqibiao['日期'].dt.strftime("%Y"),
                           年度季度 = riqibiao['日期'].dt.strftime("%Y") + "Q"+ ((riqibiao['日期'].dt.strftime("%m").astype("int32")-1)//3 + 1).astype(str),
                           年度月份 = riqibiao['日期'].dt.strftime("%Y") + riqibiao['日期'].dt.strftime("%m"))
