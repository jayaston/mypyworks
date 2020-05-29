# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:17:16 2020

@author: XieJie
"""

#连接tjfx数据库

import tjfxdata as tjfx
import bumendata as bumen
a=tjfx.TjfxData()
test = a.get_all_quota()
a.close()
b=bumen.BumenData()
b.import
# def fresh_Quota_Define():        
    
    
        
    
    

    