﻿# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import matplotlib.pyplot as plt
import sys
import os
os.getcwd()
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')
import pandas as pd
import numpy as np
import tjfxdata as tjfx

#import re    
import datetime as dt

#耗水30976，稽查24299，每月发单23765，每月追收24293，每月剔除24294，两月发单23594，两月追收23595，两月剔除23596
start = '20230101'
end = '20230131'
#list1每个单位的指标务必要完全一致。否则会出现冲突。不一致的指标放到list2
list1 = [#公司         
         ['00','00409','m'],#售水量
         ['00','23594','m'],#两月发单    
         ['00','23765','m'],#每月发单
         ['00','23426','m'],#追收水量
         ['00','23427','m'],#剔除水量
         ['00','24299','m'],#稽查水量
         ['00','30976','m'],#耗水量
         ['00','24298','m'],#二次保洁
                 
         #东山         
         ['01','00409','m'],#售水量
         ['01','23594','m'], #两月发单    
         ['01','23765','m'],#每月发单
         ['01','23426','m'],#追收水量
         ['01','23427','m'],#剔除水量
         ['01','24299','m'],#稽查水量
         ['01','30976','m'],#耗水量
         ['01','24298','m'],#二次保洁
         
         #越秀
         ['02','00409','m'],#售水量
         ['02','23594','m'], #两月发单    
         ['02','23765','m'],#每月发单
         ['02','23426','m'],#追收水量
         ['02','23427','m'],#剔除水量
         ['02','24299','m'],#稽查水量
         ['02','30976','m'],#耗水量
         ['02','24298','m'],#二次保洁
         
         #荔湾
         ['03','00409','m'],#售水量
         ['03','23594','m'], #两月发单    
         ['03','23765','m'],#每月发单
         ['03','23426','m'],#追收水量
         ['03','23427','m'],#剔除水量
         ['03','24299','m'],#稽查水量
         ['03','30976','m'],#耗水量
         ['03','24298','m'],#二次保洁
         
         #海珠
         ['04','00409','m'],#售水量
         ['04','23594','m'], #两月发单    
         ['04','23765','m'],#每月发单
         ['04','23426','m'],#追收水量
         ['04','23427','m'],#剔除水量
         ['04','24299','m'],#稽查水量
         ['04','30976','m'],#耗水量
         ['04','24298','m'],#二次保洁
         
         #芳村
         ['05','00409','m'],#售水量
         ['05','23594','m'], #两月发单    
         ['05','23765','m'],#每月发单
         ['05','23426','m'],#追收水量
         ['05','23427','m'],#剔除水量
         ['05','24299','m'],#稽查水量
         ['05','30976','m'],#耗水量
         ['05','24298','m'],#二次保洁
         
         #黄埔
         ['06','00409','m'],#售水量
         ['06','23594','m'], #两月发单    
         ['06','23765','m'],#每月发单
         ['06','23426','m'],#追收水量
         ['06','23427','m'],#剔除水量
         ['06','24299','m'],#稽查水量
         ['06','30976','m'],#耗水量
         ['06','24298','m'],#二次保洁
         
         #白云
         ['07','00409','m'],#售水量
         ['07','23594','m'], #两月发单    
         ['07','23765','m'],#每月发单
         ['07','23426','m'],#追收水量
         ['07','23427','m'],#剔除水量
         ['07','24299','m'],#稽查水量
         ['07','30976','m'],#耗水量
         ['07','24298','m'],#二次保洁
         
         #天河
         ['08','00409','m'],#售水量
         ['08','23594','m'], #两月发单    
         ['08','23765','m'],#每月发单
         ['08','23426','m'],#追收水量
         ['08','23427','m'],#剔除水量
         ['08','24299','m'],#稽查水量
         ['08','30976','m'],#耗水量
         ['08','24298','m'],#二次保洁
         
         ]

list2 = [#公司
         ['00','00718','m'],#供水总量
         ['00','00469','m'],#免费水量 
         ['00','02150','m'],#售水天数
         ['00','31594','m'],#抄表到户水量
         ['00','31512','m'],#公共管网漏损率
         #一级分区供水量
         ['0901','30984','m'],#中区
         ['0900','30984','m'],#东区
         ['0902','30984','m'],#南区
         ['0903','30984','m'],#北区
         #免费供水量
         ['0901','00469','m'],#中区
         ['0900','00469','m'],#东区
         ['0902','00469','m'],#南区
         ['0903','00469','m'],#北区
         #售水天数
         ['0901','02150','m'],#中区
         ['0900','02150','m'],#东区
         ['0902','02150','m'],#南区
         ['0903','02150','m'],#北区
         ]
shuju_df = tjfx.TjfxData().getdata(start,end,list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)


test = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')

test1 = test.stack(0)
test1.info()
test1.replace(np.nan,0,inplace = True)
test1.eval("""
           净追收水量=追收水量合计-剔除水量合计
           在册水量=两月发单水量合计+每月发单水量合计+净追收水量
           工程施工耗水 = 非水费类水合计 - 二次供水设施保洁
           """,inplace=True)
test1.drop(['追收水量合计','剔除水量合计'],axis=1,inplace=True)
test1.rename(columns= {'两月发单水量合计':'两月发单','二次供水设施保洁':'二次保洁耗水',
                       '净水售水量':'售水量','每月发单水量合计':'每月发单',
                       '违章追收水量':'稽查水量','非水费类水合计':'耗水量'},inplace=True)
test2 = test1.unstack().stack(0)
test2.eval('''
            中区分公司 = 东山片+越秀片+荔湾片
            东区分公司  = 黄埔片+天河片
            南区分公司  = 海珠片+芳村片
            北区分公司  = 白云片         
            ''',inplace=True)
test2 = test2.reindex(columns = ['广州自来水公司','中区分公司','东区分公司','南区分公司','北区分公司'])
test2 = test2.unstack()

shuju_df = tjfx.TjfxData().getdata(start,end,list2)
shuju_df.info()
shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
test3 = pd.pivot_table(shuju_df,index = ['QUOTA_DATE'],columns = ['GROUP_NAME','QUOTA_NAME'],values='QUOTA_VALUE')
test3.columns

test3.rename(columns= {'免费供水量':'免费水量',
                       '水厂供水总量':'供水总量',
                       '公共管网漏损率18年修正':'公共管网漏损率',
                       '居民用水按户抄表的水量':'抄表到户水量'},level=1,inplace=True)

test4 = pd.concat([test3,test2],axis=1, ignore_index=False)
test5=test4.reindex(columns =[
        ('广州自来水公司', '供水总量'),
        ('广州自来水公司',    '售水量'),
        ('广州自来水公司',    '在册水量'),
        ('广州自来水公司',   '两月发单'),
        ('广州自来水公司',   '每月发单'),            
        ('广州自来水公司',  '净追收水量'),            
        ('广州自来水公司',   '稽查水量'),
        ('广州自来水公司',    '耗水量'),
        ('广州自来水公司',    '工程施工耗水'),
        ('广州自来水公司',    '二次保洁耗水'),
        ('广州自来水公司',   '免费水量'),  	
        ('广州自来水公司',   '售水天数'),
        ('广州自来水公司',   '抄表到户水量'),
        ('广州自来水公司',   '公共管网漏损率'),
        (  '中区分公司', '分区供水总量'), 
        (  '中区分公司',    '售水量'), 
        (  '中区分公司',    '在册水量'),
        (  '中区分公司',   '两月发单'),
        (  '中区分公司',   '每月发单'),
        (  '中区分公司',  '净追收水量'),
        (  '中区分公司',   '稽查水量'),
        (  '中区分公司',    '耗水量'),
        (  '中区分公司',    '工程施工耗水'),
        (  '中区分公司',    '二次保洁耗水'),
        (  '中区分公司',   '免费水量'),
        (  '中区分公司',   '售水天数'),             
        (  '东区分公司', '分区供水总量'),
        (  '东区分公司',    '售水量'),
        (  '东区分公司',    '在册水量'),            
        (  '东区分公司',   '两月发单'),
        (  '东区分公司',   '每月发单'),           
        (  '东区分公司',  '净追收水量'), 
        (  '东区分公司',   '稽查水量'),
        (  '东区分公司',    '耗水量'),
        (  '东区分公司',    '工程施工耗水'),
        (  '东区分公司',    '二次保洁耗水'),
        (  '东区分公司',   '免费水量'),
        (  '东区分公司',   '售水天数'),
        (  '南区分公司', '分区供水总量'),
        (  '南区分公司',    '售水量'), 
        (  '南区分公司',    '在册水量'),
        (  '南区分公司',   '两月发单'),  
        (  '南区分公司',   '每月发单'),   
        (  '南区分公司',  '净追收水量'),
        (  '南区分公司',   '稽查水量'),
        (  '南区分公司',    '耗水量'),
        (  '南区分公司',    '工程施工耗水'),
        (  '南区分公司',    '二次保洁耗水'),
        (  '南区分公司',   '免费水量'),
        (  '南区分公司',   '售水天数'),
        (  '北区分公司', '分区供水总量'),
        (  '北区分公司',    '售水量'),
        (  '北区分公司',    '在册水量'),
        (  '北区分公司',   '两月发单'),
        (  '北区分公司',   '每月发单'),
        (  '北区分公司',  '净追收水量'),
        (  '北区分公司',   '稽查水量'),
        (  '北区分公司',    '耗水量'),
        (  '北区分公司',    '工程施工耗水'),
        (  '北区分公司',    '二次保洁耗水'),
        (  '北区分公司',   '免费水量'),
        (  '北区分公司',   '售水天数'),
        ])
test5.to_excel(r'C:\Users\XieJie\mypyworks\输出\2023年1月售水量明细水量.xlsx')
#list1 = [x+'_'+y for x,y in zip(test.columns.get_level_values(0).values , test.columns.get_level_values(1).values)]  
#test.columns = list1
#test6=test5.swaplevel(axis=1)[['售水量','分区供水总量']].resample('Y').sum()

