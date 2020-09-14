# -*- coding: utf-8 -*-
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
#水量主题数据集
shuiliang_df = shuju_df[
    shuju_df['QUOTA_DEPT_CODE'].isin(['1001','1002','100301','100302','1004','1005','1007','1016','1009','00']) & 
    shuju_df['QUOTA_CODE'].isin(['11930','04281','29860','20640','30964','29861','00700','29862','00718','00752','04346','04347','30985','02380'])
    ].query("RECORD_TYPE=='d'")
shuiliang_df['QUOTA_DATE'] = shuiliang_df['QUOTA_DATE'].dt.strftime("%Y-%m-%d")
shuiliang_df.QUOTA_VALUE = pd.to_numeric(shuiliang_df.QUOTA_VALUE,errors='coerce')
shuiliang_df = shuiliang_df[np.isfinite(shuiliang_df.QUOTA_VALUE)]

shuiliang_df = pd.pivot_table(shuiliang_df,
                              index = ['QUOTA_DATE','GROUP_NAME'],
                              columns = ['QUOTA_NAME'],values='QUOTA_VALUE',
                              fill_value=0).reset_index()
del shuju_df




