# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:19:57 2019

@author: Jay
"""
import sys
sys.path.append(r'C:\Users\XieJie\mypyworks\StatLedger\module')
import pandas as pd
import numpy as np
#import re    
import datetime as dt
from bumendata import BumenData

enddate = (dt.datetime.now()-dt.timedelta(days=1)).strftime('%Y%m%d')
startdate = (dt.datetime.now()-dt.timedelta(days=731)).strftime('%Y%m%d')

riqibiao = pd.DataFrame({'日期':pd.date_range(startdate,enddate)})
riqibiao = riqibiao.assign(年度 = riqibiao['日期'].dt.strftime("%Y"),
                           年度季度 = riqibiao['日期'].dt.strftime("%Y") + "Q"+ ((riqibiao['日期'].dt.strftime("%m").astype("int32")-1)//3 + 1).astype(str),
                           年度月份 = riqibiao['日期'].dt.strftime("%Y") + riqibiao['日期'].dt.strftime("%m"))

shuju_df = BumenData().getdata(startdate,enddate)
shuju_df = shuju_df.query("RECORD_TYPE!='h'")
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce')
shuju_df = shuju_df[np.isfinite(shuju_df.QUOTA_VALUE)]

selectquota = pd.read_excel(r"C:\Users\XieJie\mypyworks\StatLedger\数据表\数据集_关键指标看板.xlsx",
                            dtype={'QUOTA_CODE':object},
                            usecols=['QUOTA_CODE'])
selectquotalist = list(set(selectquota['QUOTA_CODE']))

shuju_df = shuju_df[shuju_df['QUOTA_CODE'].isin(selectquotalist)]

shuju_df = pd.pivot_table(shuju_df,index = ['QUOTA_DATE','GROUP_NAME','RECORD_TYPE'],
                      columns = ['QUOTA_NAME'],values='QUOTA_VALUE',fill_value=0)
shuju_df = shuju_df.reset_index()
shuju_df = shuju_df.rename(columns= {'取水量':'对外取水量','入厂取水量':'取水量'})
