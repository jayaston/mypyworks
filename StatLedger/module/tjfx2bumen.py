# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:17:16 2020

@author: XieJie
"""

#连接tjfx数据库
import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')
import tjfxdata as tjfx
import bumendata as bumen

test = tjfx.TjfxData().get_all_quota()

bumen.BumenData().importdata()

# def fresh_Quota_Define():        
    
    
        
    
    

    