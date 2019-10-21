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
list1 = [['1001','00718','m'],
         ['1002','00718','m'],
         ['1003','00718','m'],
         ['100301','00718','m'],
         ['100302','00718','m'],
         ['1004','00718','m'],
         ['1005','00718','m'],
         ['1006','00718','m'],
         ['1007','00718','m']
         ]
shuju_df = tjfx.TjfxData().getdata('20090101','20181231',list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)
shuju_df = pd.pivot_table(shuju_df,index='QUOTA_DATE',columns=['GROUP_NAME'],values=['QUOTA_VALUE'])

result = shuju_df.resample('Y').sum()




result.to_excel("E:\\pyworks\\近十年各厂供水总量.xls")
