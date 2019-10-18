# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:17:32 2019

@author: XieJie
"""
import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\StatLedger\module')
import cumcalculate as cc
import tjfxdata as tjfx

startd,endd,quotalist = '20190101','20191009',['d_00_31195','d_1009_31195','d_1001_31195','d_1002_31195',
                                               'd_1003_31195','d_1016_31195','d_1004_31195','d_1005_31195',
                                               'd_1007_31195']
result = cc.all_calcu(startd,endd,quotalist)