# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:17:16 2020

@author: XieJie
"""

import sys
import os
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"StatLedger\module")))
except:
    sys.path.append(r'.\mypyworks\StatLedger\module')

from bumendata import BumenData
from tjfxdata import TjfxData
import pandas as pd
import numpy as np
#同步指标信息表
#提取指标表
# def fresh_Quota_Define():  
df_quota = TjfxData().get_all_quota()
#写入指标表
BumenData().imp_tbl(df_quota,'CS_QUOTA_DEFINE')

#同步部门信息表
#提取部门表
df_dept = TjfxData().get_all_dept()
#写入部门表
BumenData().imp_tbl(df_dept,'HR_ORGANIZATION') 


#同步公式信息表
#提取本地公式
#提取系统公式表
formulaset = TjfxData().get_formula().query("TZ_TYPE != 'd' ")#提取公式概要表
formulaset = formulaset.rename(columns = {'QUOTA_NAME':'zbming','QUOTA_CODE':'var_code','ZB_DEPT_CODE':'var_dept','TZ_TYPE':'RECORD_TYPE'})
formulaset = formulaset.reindex(columns=['zbming','var_code','var_dept','RECORD_TYPE','START_TIME','END_TIME','FORMULA_CODE'])
#提取公式详情表
formuladetail=TjfxData().get_formula_detail() 
formuladetail = formuladetail[formuladetail['FORMULA_CODE'].isin( set(formulaset['FORMULA_CODE']))]
formuladetail.sort_values(['FORMULA_CODE','FLOW_NO'],inplace=True)
formuladetail = formuladetail.fillna(value='')#填充空值
formuladetail['PARAMETER']=formuladetail['PARAMETER'].astype('str')#参数列改成字符型
#改变计算符号
formuladetail.replace({'OPERATION':{"0":"+","1":"-","2":"*","3":"/","4":"",
                          "5":"MaxHNum","6":"MaxH","7":"MinHNum","8":"MinH",
                          "9":"MaxDNum","10" :"MaxD","11":"MinDNum","12":"MinD"}},inplace=True)
#插入空格和-
col_name = formuladetail.columns.tolist()
col_name.insert(col_name.index('QUOTA_CODE'),'lianjie')
formuladetail=formuladetail.reindex(columns=col_name,fill_value='_')
col_name = formuladetail.columns.tolist()
col_name.insert(col_name.index('QUOTA_DEPT_CODE'),'space1')
col_name.insert(col_name.index('RIGHT_BRACKET'),'space2')
formuladetail=formuladetail.reindex(columns=col_name,fill_value=' ')


#移除汉和特殊字符
def removeChnAndCharacter(str1):
    #将中文标点符号转换为英文标点符号
    def C_trans_to_E(string):
        E_pun = u',.!?[]()<>"\''
        C_pun = u'，。！？【】（）《》“‘'
        #ord返回ASCII码对应的int
        #zip将合并为列表，元素为元祖，元祖为对应位置所有元素依次的集合，如这种形式[(',','，')...]
        #s生成对应字典
        table= {ord(f):ord(t) for f,t in zip(C_pun,E_pun)}
        #将字符传对应转换
        return string.translate(table)
    C_pun = u'，。！？【】（）《》“‘'
    strTmp = ''
    if not isinstance(str1,str):
        return strTmp
    for i in range(len(str1)):
        #中文字符范围
        #https://blog.csdn.net/qq_22520587/article/details/62454354
        if str1[i] >= u'\u4e00' and str1[i] <= u'\u9fa5' \
                or str1[i] >= u'\u3300' and str1[i] <= u'\u33FF' \
                or str1[i] >= u'\u3200' and str1[i] <= u'\u32FF' \
                or str1[i] >= u'\u2700' and str1[i] <= u'\u27BF' \
                or str1[i] >= u'\u2600' and str1[i] <= u'\u26FF' \
                or str1[i] >= u'\uFE10' and str1[i] <= u'\uFE1F' \
                or str1[i] >= u'\u2E80' and str1[i] <= u'\u2EFF' \
                or str1[i] >= u'\u3000' and str1[i] <= u'\u303F' \
                or str1[i] >= u'\u31C0' and str1[i] <= u'\u31EF' \
                or str1[i] >= u'\u2FF0' and str1[i] <= u'\u2FFF' \
                or str1[i] >= u'\u3100' and str1[i] <= u'\u312F' \
                or str1[i] >= u'\u21A0' and str1[i] <= u'\u31BF' \
                :
            pass#中文字符不处理
        else:
            if str1[i] in C_pun:
                # st = C_trans_to_E(str1[i])#中文标点不处理
                pass
            else:
                st = str1[i]
            strTmp += st
    return strTmp

formuladetail['QUOTA_DEPT_CODE']= formuladetail['QUOTA_DEPT_CODE'].apply(removeChnAndCharacter)

formuladetail.info()


#构造公式表达式函数按
def get_formula(df):
    nrows=len(df)
    formula_exp = ''
    for i in range(nrows):
        for j in range(1,10) :            
            formula_exp += df.iat[i,j]
    return formula_exp 
        
test = formuladetail.groupby(['FORMULA_CODE']).apply(get_formula)
test = test.str.replace(' _ ','')
test=pd.DataFrame(test)
test.columns=['setformula']                
test.reset_index(inplace=True)  

#合并系统公式表


#显示冲突公式
test1 = test[test.duplicated(['TZ_TYPE','QUOTA_CODE','ZB_DEPT_CODE'],keep= False)]\
    .sort_values(['TZ_TYPE','QUOTA_CODE','ZB_DEPT_CODE'])

#选取同一个指标的第一个公式
test2 = test.drop_duplicates(['TZ_TYPE','QUOTA_CODE','ZB_DEPT_CODE'],keep='first')
#验证是否还有同一个指标
test2[test2.duplicated(['TZ_TYPE','QUOTA_CODE','ZB_DEPT_CODE'],keep= False)]
test2.info()
test2 = test2.rename(columns = {'QUOTA_NAME':'zbming','QUOTA_CODE':'var_code','ZB_DEPT_CODE':'var_dept','TZ_TYPE':'RECORD_TYPE'})
test2 = test2.reindex(columns=['zbming','var_code','var_dept','RECORD_TYPE','START_TIME','START_TIME','FORMULA_CODE'])

len()

#提取公式细节















#提取公式的开始和结束时间

formula_time = sorted(set(pd.concat((test2['START_TIME'],test2['END_TIME']),axis=0)))
formula_time.pop(0)

#提取时间指标
df = TjfxData().getdata('20200701','20200720')
df_h = df.query("RECORD_TYPE=='h'")

#时指标公式校验并提示错误

#时指标写入数据表

#向上聚合成日指标


#与日指标合并提示差异

#日指标公式校验并提示错误数据

#日指标写入数据库

#向上聚合成月指标



#和月指标合并并提示差异

#月指标公式校验并提示错误

#月指标写入数据库

#向上汇聚成年指标

#写如mysql数据库



    
    
        
    
    

    