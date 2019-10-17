# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:19:57 2019

@author: Jay
"""

import sys
sys.path.append(r'C:\Users\Jay\mypyworks\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx
#import re    
import datetime as dt
nowtime = dt.datetime.now().strftime('%Y%m%d')
startdate = (dt.datetime.now()-dt.timedelta(days=731)).strftime('%Y%m%d')

shuju_df = tjfx.TjfxData().getdata(startdate,nowtime)
shuju_df = shuju_df.query("RECORD_TYPE!='h'")
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)
selectquota = pd.read_excel(r"C:\Users\Jay\mypyworks\StatLedger\数据表\水厂看板指标.xlsx",
                            dtype={'QUOTA_CODE':object,'QUOTA_NAME':object},
                            usecols=['QUOTA_CODE'])

shuju_df = pd.merge(shuju_df,selectquota,on="QUOTA_CODE")
shuju_df = pd.pivot_table(shuju_df,index = ['QUOTA_DATE','GROUP_NAME','RECORD_TYPE'],
                      columns = ['QUOTA_NAME'],values='QUOTA_VALUE',fill_value=0)
shuju_df = shuju_df.reset_index()
shuju_df.info()
