# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:00:13 2019

@author: XieJie
"""
import sys
sys.path.append(r'E:\pyworks\StatLedger\module')
import pandas as pd
import numpy as np
from tjfxdata import TjfxData
import re   
import cx_Oracle


def order_gongshiku():
        gongshiku = pd.read_excel('C:\\Users\\XieJie\\Desktop\\tjfx数据库\\python公式库北部.xlsx','layer3',dtype='object')
    
        gongshiku = gongshiku.drop(['zbming'],axis=1)
        
        gongshiku['zhibiao'] = gongshiku['RECORD_TYPE']+'_'+gongshiku['var_dept']+'_'+gongshiku['var_code']
            
        
        def newformula(x):#对通用公式进行还原函数
            result = re.sub(r'(?=\b_)', x['var_dept'],x['setformula'])
            result = re.sub(r'(?<=_\b)', x['var_code'],result)
            result = re.sub(r'(?=\b\d+_)', x['RECORD_TYPE']+'_',result)
            return result
        
        gongshiku['setformula'] = gongshiku.apply(newformula,axis=1)
        gongshiku['zhibiaoji'] = gongshiku.apply(lambda x: re.findall(r'\b[a-z]_\d+_\d+\b',x['setformula']),axis=1)
        gongshiku.drop(['var_code','var_dept'],axis=1,inplace=True)
        
        def insert_sort(ilist,idict):#对公式库进行排序函数
            for i in range(len(ilist)):
                for j in range(i):
                    if ilist[i] in idict[ilist[j]]:
                        ilist.insert(j,ilist.pop(i))
                        break
            return ilist
        
        gongshiku_dict = dict(zip(gongshiku['zhibiao'],gongshiku['zhibiaoji']))
        #gongshiku_dict2 = gongshiku[['zhibiao','zhibiaoji']].set_index('zhibiao').T.to_dict('list')
        zhibiao_list = list(gongshiku['zhibiao'])
        zhibiao_list = insert_sort(zhibiao_list,gongshiku_dict)
         
        zhibiao_list = sorted(set(zhibiao_list),key=zhibiao_list.index) 
        
        gongshiku['formula']=gongshiku['zhibiao']+' = '+gongshiku['setformula']
        #gongshiku.set_index('zhibiao',inplace=True)
        
        #此处重写排序公式
        gongshiku['zhibiao'] = gongshiku['zhibiao'].astype('category')
        gongshiku['zhibiao'].cat.reorder_categories(zhibiao_list, inplace=True)
        gongshiku['theindex'] = list(gongshiku.index)
        gongshiku.sort_values(['zhibiao','theindex'], inplace=True)
        gongshiku = gongshiku.reset_index()     
        
        #gongshiku.to_excel("E:\\pyworks\\gongshiku.xls")
        #gongshiku=gongshiku.reindex(zhibiao_list)
        gongshiku.to_excel(r"C:\Users\XieJie\Desktop\排序公式库.xlsx")
        return gongshiku
    


startd,endd = ('20190701','20190831')
def accum_tjfx(startd,endd):
    gongshiku = order_gongshiku()    
    shuju_df = TjfxData().getdata(startd,endd)
#    shuju_df.to_excel(r'E:\pyworks\输出\测试数据源.xlsx')
    shuju_df = shuju_df.query("RECORD_TYPE=='d'")
    
    shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coercs').fillna(0)
    shuju_df.info()
    shuju_df_dcast = pd.pivot_table(shuju_df,index='QUOTA_DATE',
                                columns=['RECORD_TYPE','QUOTA_DEPT_CODE','QUOTA_CODE'],
                                values = 'QUOTA_VALUE' )    
     
    new_colnames = ["_".join(list(i)) for i in list(shuju_df_dcast.columns)]    
    shuju_df_dcast.columns=new_colnames
#    shuju_df_dcast[list(set(sum(list(gongshiku['zhibiaoji']),[])))]
    
    zhibiaoall = list(set(re.findall(r'\b[a-z]_\d+_\d+\b',' '.join(list(gongshiku['setformula'])))))
    zhibiaoall.extend(new_colnames)
    zhibiaoall = list(set(zhibiaoall))
    
    shuju_df_dcast = shuju_df_dcast.reindex(columns=zhibiaoall, fill_value=0)
    
    test = shuju_df_dcast.groupby(pd.Grouper(freq="M")).agg()
    agg_dict = {key:    for key in zhibiaoall }
    
    
    
    
    
    
    
    
#    shuju_df_dcast_n.to_excel(r'E:\pyworks\输出\测试数据源.xlsx')
    
    
    
    
    
    
    
    shuju_leiji = shuju_df.groupby(['RECORD_TYPE','QUOTA_DEPT_CODE','QUOTA_CODE']).sum()
    
    shuju_leiji['zhibiao'] =list( pd.Series(shuju_leiji.index.values).apply(lambda x: "_".join(x)))
    
    #shuju_leiji['zhibiaomingcheng'] =list( pd.Series(shuju_leiji.index.values).apply(lambda x: "_".join(x[3:])))
    
    shuju_leijijisuan = shuju_leiji.reset_index().query("QUOTA_CODE != '02380'")[['zhibiao','QUOTA_VALUE']]
    
    shuju_leijijisuan = pd.concat([shuju_leijijisuan,shuju_df_ave])
    zhibiaoall_df = pd.DataFrame({'zhibiao':list(set(re.findall(r'\b[a-z]_\d+_\d+\b',' '.join(list(gongshiku['setformula'])))))})
    shuju_leijijisuan = pd.merge(zhibiaoall_df,shuju_leijijisuan,on='zhibiao',how='outer')
    shuju_leijijisuan['QUOTA_VALUE'] = pd.to_numeric(shuju_leijijisuan['QUOTA_VALUE'],errors='coercs').fillna(0)
    shuju_leijijisuan_t = shuju_leijijisuan.set_index('zhibiao').T
    
    
    
    
    for item in gongshiku['formula'].map(lambda x:x.replace(' ','')):    
        shuju_leijijisuan_t.eval(item,inplace=True)   
    #shuju_leijijisuan_t.to_excel(r"C:\Users\XieJie\Desktop\计算结果.xlsx")
    #此处应该用数据库查找指标名称。下行修改。
    sql2 = "SELECT DISTINCT QUOTA_CODE, QUOTA_NAME \
                FROM zls_tjfx.cs_Quota_Define" 
    sql3 = "SELECT DISTINCT GROUP_CODE , GROUP_NAME \
                FROM zls_tjfx.hr_organization" 
    conn = cx_Oracle.connect("zls_tjfx/tjfx10goracle@10.1.12.196:1521/orcl")
    zhibiao_df = pd.read_sql(sql2,conn)
    bumen_df = pd.read_sql(sql3,conn)
    conn.close()
    
    shuju_leijijisuan = shuju_leijijisuan_t.T.reset_index()
    shuju_leijijisuan['QUOTA_CODE'] = shuju_leijijisuan['zhibiao'].map(lambda x: x.split('_')[2])
    shuju_leijijisuan['GROUP_CODE'] = shuju_leijijisuan['zhibiao'].map(lambda x: x.split('_')[1])
    
    shuju_leiji = pd.merge(shuju_leijijisuan,zhibiao_df,how='left',on='QUOTA_CODE')
    shuju_leiji = pd.merge(shuju_leiji,bumen_df,how='left',on='GROUP_CODE').rename(columns={"GROUP_CODE":"QUOTA_DEPT_CODE"})
    
    shuju_leiji = shuju_leiji[shuju_leiji['QUOTA_VALUE'].notna()]
    
    shuju_leiji = shuju_leiji.set_index('zhibiao').sort_index(axis=0)
    
    shuju_leiji.to_excel(r"C:\Users\XieJie\Desktop\累计计算结果.xlsx")
    return shuju_leiji

if __name__=="__main__":
    test = accum_tjfx("20190201","20190228")



