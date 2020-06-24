# -*- coding: utf-8 -*-
import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')

import pandas as pd
import numpy as np
import tjfxdata as tjfx
#import re  
list1 = [['1001','00718','d'],
         ['1002','00718','d'],
         ['1003','00718','d'],
         ['100301','00718','d'],
         ['100302','00718','d'],
         ['1004','00718','d'],
         ['1005','00718','d'],
         
         ['1007','00718','d']
         #['1016','00718','d']
         ]
shuju_df = tjfx.TjfxData().getdata('20160101','20181231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
shuju_df1 = pd.pivot_table(shuju_df,index='QUOTA_DATE',columns=['GROUP_NAME'],values= 'QUOTA_VALUE')
shuju_df1 = shuju_df1.reset_index()

type(shuju_df1.iloc[:,1] )

    
#按年最大值
get_max = lambda x: x[x.iloc[:,1]==x.iloc[:,1].max()]
                        
# python就是灵活啊。
get_max.__name__ = "maxday"

test = dict()
for i in list(shuju_df1.columns)[1:]:
    item = shuju_df1[['QUOTA_DATE',i]].groupby([pd.Grouper(key='QUOTA_DATE',freq='Y')]).apply(get_max)
    test[i] = item


result.to_excel("E:\\pyworks\\近十年各厂供水总量.xls")
