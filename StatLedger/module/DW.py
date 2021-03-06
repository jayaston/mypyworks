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
import os
import datetime as dt
import re
os.getcwd()

#同步公式信息表、目录指标关系表、方案目录关系表、方案表等
for tbl in ['cs_formula_set','Cs_Formula_Detail','CS_TZZB_RELATION','CS_TZ_ITEM','CS_ITEM_VIEW','CS_TZ_VIEW','CS_VIEW_ACCOUNTS']:
    
    df = TjfxData().get_any_data(sql = 'select * from '+ tbl)
    BumenData().imp_tbl(df,tbl)

#同步指标表
df_quota = TjfxData().get_all_quota()
df_quota.to_excel('指标.xls')
BumenData().imp_tbl(df_quota,'CS_QUOTA_DEFINE')

#同步部门表
df_dept = TjfxData().get_all_dept()
BumenData().imp_tbl(df_dept,'HR_ORGANIZATION') 



#提取系统公式
formulaset = TjfxData().get_formula()#.query("TZ_TYPE != 'd' ")
formulaset = formulaset.rename(columns = {'QUOTA_NAME':'zbming','QUOTA_CODE':'var_code','ZB_DEPT_CODE':'var_dept','TZ_TYPE':'RECORD_TYPE'})
formulaset = formulaset.reindex(columns=['zbming','var_code','var_dept','RECORD_TYPE','START_TIME','END_TIME','FORMULA_CODE','FORMULA','方案','目录'])
#len(formulaset['FORMULA_CODE'].unique())
#提取公式详情表
formuladetail=TjfxData().get_formula_detail() 
formuladetail = formuladetail[formuladetail['FORMULA_CODE'].isin( set(formulaset['FORMULA_CODE']))]
formuladetail.sort_values(['FORMULA_CODE','FORMULA_ORDER_SN','FLOW_NO'],inplace=True)#增加排序字段，第115行代码选择公式的范围也需要修改
formuladetail = formuladetail.fillna(value='')#填充空值
formuladetail['PARAMETER']=formuladetail['PARAMETER'].astype('str')#参数列改成字符型
#改变计算符号
formuladetail.replace({'OPERATION':{"0":"+","1":"-","2":"*","3":"/","4":"",
                          "5":"MaxHNum","6":"MaxH","7":"MinHNum","8":"MinH",
                          "9":"MaxDNum","10" :"MaxD","11":"MinDNum","12":"MinD",
                          "13":"前n期"}},inplace=True)
#插入空格和-
col_name = formuladetail.columns.tolist()
col_name.insert(col_name.index('QUOTA_CODE'),'lianjie')
formuladetail=formuladetail.reindex(columns=col_name,fill_value='_')
col_name = formuladetail.columns.tolist()
col_name.insert(col_name.index('QUOTA_DEPT_CODE'),'space1')
col_name.insert(col_name.index('RIGHT_BRACKET'),'space2')
formuladetail=formuladetail.reindex(columns=col_name,fill_value=' ')

#移除汉和特殊字符函数
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
#构造公式表达式函数按
def get_formula(df):
    nrows=len(df)
    formula_exp = ''
    for i in range(nrows):
        for j in range(2,10) :            
            formula_exp += df.iat[i,j]
    return formula_exp 
        
formuladetail = formuladetail.groupby(['FORMULA_CODE']).apply(get_formula)
formuladetail = formuladetail.str.replace(' _ ','')
formuladetail=pd.DataFrame(formuladetail)
formuladetail.columns=['setformula']                
formuladetail.reset_index(inplace=True)  

#系统公式表
formula_tjfx = pd.merge(formulaset,formuladetail,how='left',on=['FORMULA_CODE'])
formula_tjfx = formula_tjfx.reindex(columns=['zbming', 'var_code', 'var_dept', 'RECORD_TYPE', 'START_TIME',
       'END_TIME', 'FORMULA_CODE', 'setformula','FORMULA', '方案', '目录'])

#提取本地公式
# dir = os.path.dirname(os.path.dirname(__file__))
# path = os.path.abspath(os.path.join(dir,"数据表/全新公式库北部太和.xlsx"))    
formula_excel = pd.read_excel(r'./mypyworks/StatLedger/数据表/新格式公式表.xlsx',
                          sheet_name='formula',dtype={'var_code':str,'var_dept':str})


formula_DH = pd.merge(formula_tjfx.query(
    "RECORD_TYPE=='d' | RECORD_TYPE=='h'").drop_duplicates(
        ['var_code','var_dept','RECORD_TYPE'])[['zbming','var_code','var_dept','RECORD_TYPE','方案','目录']],
                 formula_excel.drop('zbming',axis=1),how='right', #正常用right，如果查找全部指标用outer
                 on=['var_code','var_dept','RECORD_TYPE'])

formula_M = formula_tjfx.query("RECORD_TYPE=='m'")

#系统公式表与本地公式表合并
formula = pd.concat([formula_M,formula_DH],ignore_index=True).reindex(
    columns=['zbming','var_code','var_dept','RECORD_TYPE','START_TIME','END_TIME','setformula','FORMULA_CODE','FORMULA','方案','目录'])


#注意此委提取所有指标并导出可能有的公式并不是主程序，在tjfxdata中修改了sql语句才使用。
# formula.rename(
#     columns={'zbming':'指标名','var_code':'指标编码','var_dept':'部门编码','RECORD_TYPE':'指标类型',
#               'START_TIME':'公式有效期开始','END_TIME':'公式有效期结束','setformula':'公式表达式','FORMULA_CODE':'公式编码','FORMULA':'公式注释'}).to_excel(
#     './mypyworks/输出/有效方案中所有指标以及公式.xls')

#写入mysql数据库
BumenData().imp_tbl(formula,'FORMULA') 



#同步数据表
sql="select t.*\
    from zls_tjfx.tj_quota_data t \
        where t.quota_date >= to_date('19900101','yyyymmdd')\
           and t.quota_date < to_date('20201110','yyyymmdd')\
             and to_char(quota_date,'ss')='00'\
             and t.quota_value != '0'\
             and t.quota_value is not null\
             and length(t.QUOTA_DEPT_CODE) = lengthb(t.QUOTA_DEPT_CODE)"
df = TjfxData().get_any_data(sql = sql)

#df转换为list
# df = df.reindex(columns=['QUOTA_CODE','MON','QUOTA_DATE','QUOTA_VALUE',
#                              'REPORT_FLAG','QUOTA_DEPT_CODE','IMPORT_FLOW_NO',
#                              'WARNING_CODE','RECORD_TYPE'])
# df['MON'] = df['QUOTA_DATE'].dt.strftime('%Y%m')
df['MON'] = df['MON'].astype('str')
df['QUOTA_DATE'] = df['QUOTA_DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')
df['QUOTA_VALUE'] = df['QUOTA_VALUE'].astype('str')
mylist = df.values.tolist()
BumenData().importdata(mylist)



    
    
        
    
    

    