# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:31:59 2019

@author: XieJie
"""
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
import rpy2.robjects as robjects
startd = '20211201'
endd =   '20211231' #不能跨年
r_script = '''
Sys.setlocale('LC_ALL', locale = "English_United States.1252") 

library(RODBC)
library(reshape2)
library(dplyr)
library(foreign)
library(readxl)
library(xlsx)
options(scipen=10)

odbcCloseAll()

startd = ''' + startd +'''
endd= '''+endd+'''

startyear<-as.numeric(substr(startd,1,4))

channel1<-odbcConnect("ShenChanBu.TianQi2",uid = "jitongbu",pwd = "xiejie")
tianqiqingkuang<-sqlQuery(channel1, paste0("select * 
                                        from AliWeather",startyear,"
                                           where (CONVERT(varchar(12) , fDate, 112 ) >=",startd,"
                                           and CONVERT(varchar(12) , fDate, 112 ) <=",endd,")
                                           and fNo in (1,4)")
                          ,stringsAsFactors = FALSE)
tianqiqingkuang.Matrix<-as.matrix(tianqiqingkuang[,-(1:2)])
tianqiqingkuang.max<-apply(tianqiqingkuang.Matrix,1,max,na.rm=TRUE)
tianqiqingkuang.min<-apply(tianqiqingkuang.Matrix,1,min,na.rm=TRUE)
tianqiqingkuang.mean<-round(apply(tianqiqingkuang.Matrix,1,mean,na.rm=TRUE),1)
tianqiqingkuang.huizong<-data.frame(tianqiqingkuang[(1:2)],tianqiqingkuang.max,tianqiqingkuang.min,tianqiqingkuang.mean)


tianqiqingkuang.huizong%>%filter(.,fNo==1)%>%select(.,-1)%>%
  rename(.,QUOTA_DATE=fDate,`12190`=tianqiqingkuang.max,`12225`=tianqiqingkuang.min,`12260`=tianqiqingkuang.mean)%>%
  melt(., id.vars = c('QUOTA_DATE'), variable.name ='QUOTA_CODE', value.name = 'QUOTA_VALUE', factorsAsStrings = FALSE)%>%
  mutate(.,QUOTA_DEPT_CODE='00',QUOTA_DATE=as.character(QUOTA_DATE,'%Y-%m-%d'),QUOTA_CODE=as.character(QUOTA_CODE))%>%
  select(.,QUOTA_DATE,QUOTA_DEPT_CODE,QUOTA_CODE,QUOTA_VALUE)->tianqiqingkuang.qiwen


tianqiqingkuang.huizong%>%filter(.,fNo==4)%>%select(.,-1)%>%
  rename(.,QUOTA_DATE=fDate,`31196`=tianqiqingkuang.max,`31197`=tianqiqingkuang.min,`31198`=tianqiqingkuang.mean)%>%
  melt(., id.vars = c('QUOTA_DATE'), variable.name ='QUOTA_CODE', value.name = 'QUOTA_VALUE', factorsAsStrings = FALSE)%>%
  mutate(.,QUOTA_DEPT_CODE='00',QUOTA_DATE=as.character(QUOTA_DATE,'%Y-%m-%d'),QUOTA_CODE=as.character(QUOTA_CODE))%>%
  select(.,QUOTA_DATE,QUOTA_DEPT_CODE,QUOTA_CODE,QUOTA_VALUE)->tianqiqingkuang.shidu

tianqi.d<-rbind(tianqiqingkuang.qiwen,tianqiqingkuang.shidu)
close(channel1)

channel2<-odbcConnect("ShenChanBu.Fenqu",uid = "jitongbu",pwd = "xiejie")
fenqushuiliang<-sqlFetch(channel2,paste("R1Day",startyear,sep = ''))
fenqushuiliang%>%filter(.,fTagNo < 0)%>%
    mutate(.,fMonth=sprintf("%02d",fMonth),fDay=sprintf("%02d",fDay),
           QUOTA_DATE=paste(startyear,fMonth,fDay,sep=''),
           QUOTA_DEPT_CODE=fTagNo,
           #QUOTA_CODE='30984',
           QUOTA_VALUE=fData)%>%
    filter(.,QUOTA_DATE>=startd&QUOTA_DATE<=endd)%>%
    select(.,-c('fTagNo','fData','fModify','fLock'))%>%
    mutate(.,QUOTA_DEPT_CODE=-QUOTA_DEPT_CODE)->fenqushuiliang.1ji  

Area_info<-sqlFetch(channel2,'Area_info')%>%filter(.,fVaild==1)

fenqushuiliang<-merge(fenqushuiliang.1ji,Area_info,
    by.x = 'QUOTA_DEPT_CODE', by.y = 'fmain',all=FALSE)

fenqushuiliang%>%dcast(fMonth+fName~fDay,value.var="QUOTA_VALUE",fill = '')->fenqushuiliang

fenqushuiliang$fName <- iconv(as.vector(fenqushuiliang$fName), 'GB2312', 'UTF-8')
close(channel2)



channel3<-odbcConnect("ShenChanBu.DunShou",uid = "jitongbu",pwd = "xiejie")
dunshou<-sqlFetch(channel3,paste("R1Day",startyear,sep = ''))

Area_info<-sqlFetch(channel3,'Area_info')
tag_info<-sqlFetch(channel3,'tag_info')
Relation_Info<-sqlFetch(channel3,'Relation_Info')


dunshou%>%filter(.,fTagNo < 0)%>%
  mutate(.,fMonth=sprintf("%02d",fMonth),fDay=sprintf("%02d",fDay),
         QUOTA_DATE=paste(startyear,fMonth,fDay,sep=''),
         QUOTA_DEPT_CODE=fTagNo,
         #QUOTA_CODE='30984',
         QUOTA_VALUE=fData)%>%
  filter(.,QUOTA_DATE>=startd&QUOTA_DATE<=endd)%>%
  select(.,-c('fTagNo','fData','fModify','fLock'))%>%
  mutate(.,QUOTA_DEPT_CODE=-QUOTA_DEPT_CODE)->dunshoushuiliang.dunshou 

dunshoushuiliang<-merge(dunshoushuiliang.dunshou ,Area_info,
                        by.x = 'QUOTA_DEPT_CODE', by.y = 'fMain',all=FALSE)

dunshoushuiliang%>%dcast(fMonth+fName~fDay,value.var="QUOTA_VALUE",fill = '')->dunshoushuiliang



Relation_Info %>% filter(fForArea %in% c(1,2)) ->Relation_Info

dunshou%>%filter(fTagNo %in% Relation_Info$fTagMain) %>%
  mutate(.,fMonth=sprintf("%02d",fMonth),fDay=sprintf("%02d",fDay),
         QUOTA_DATE=paste(startyear,fMonth,fDay,sep=''),
         QUOTA_DEPT_CODE=fTagNo,
         #QUOTA_CODE='30984',
         )%>%
  filter(.,QUOTA_DATE>=startd&QUOTA_DATE<=endd)%>%
  mutate(QUOTA_VALUE = ifelse(is.na(fData),0,fData) + ifelse(is.na(fModify),0,fModify))%>%
  select(.,-c('fTagNo','fData','fModify','fLock'))->dunshoushuiliang.dunshoumingxi 

dunshoumingxi<-merge(dunshoushuiliang.dunshoumingxi ,tag_info,
                        by.x = 'QUOTA_DEPT_CODE', by.y = 'fMain',all=FALSE)

dunshoumingxi%>%dcast(fMonth+fName~fDay,value.var="QUOTA_VALUE",fill = '')->dunshoumingxi

dunshoushuiliang$fName <- iconv(as.vector(dunshoushuiliang$fName), 'GB2312', 'UTF-8')
dunshoumingxi$fName <- iconv(as.vector(dunshoumingxi$fName), 'GB2312', 'UTF-8')
close(channel3)
'''

robjects.r(r_script)
matrix = robjects.r['fenqushuiliang']#将dataframe变量通过运行变量名调出来
matrix1 = robjects.r['tianqiqingkuang.qiwen']
matrix2 = robjects.r['dunshoushuiliang']
matrix3 = robjects.r['dunshoumingxi']
#耗水30976，稽查24299，每月发单23765，每月追收24293，每月剔除24294，两月发单23594，两月追收23595，两月剔除23596
a = np.array(matrix)#转化为二维数组
a = a.transpose()

a1 = np.array(matrix1)#转化为二维数组
a1 = a1.transpose()

a2 = np.array(matrix2)#转化为二维数组
a2 = a2.transpose()

a3 = np.array(matrix3)#转化为二维数组
a3 = a3.transpose()

test_a1= pd.DataFrame(a1,columns=['QUOTA_DATE','QUOTA_DEPT','QUOTA_NAME','QUOTA_VALUE'])
test_a1['QUOTA_DATE'] = pd.to_datetime(test_a1['QUOTA_DATE'], format='%Y-%m-%d')
test_a1.QUOTA_VALUE = pd.to_numeric(test_a1.QUOTA_VALUE,errors='coerce').fillna(0)

test_a1['mon']=test_a1['QUOTA_DATE'].dt.strftime('%m')
test_a1['day']=test_a1['QUOTA_DATE'].dt.strftime('%d')
test_a1['QUOTA_NAME'].replace(['12190','12225','12260'],['最高温','最低温','平均温'],inplace=True)

test_a1.info()
test_a1 = pd.pivot_table(test_a1,index=['mon','QUOTA_NAME'],columns='day',values='QUOTA_VALUE')
test_a1=test_a1.reset_index()
test_a1=test_a1.values


#转化趸售

# test_a2= pd.DataFrame(a2,columns=['QUOTA_DATE','QUOTA_DEPT','QUOTA_NAME','QUOTA_VALUE'])
# test_a2['QUOTA_DATE'] = pd.to_datetime(test_a2['QUOTA_DATE'], format='%Y-%m-%d')
# test_a2.QUOTA_VALUE = pd.to_numeric(test_a2.QUOTA_VALUE,errors='coerce').fillna(0)

# test_a2['mon']=test_a2['QUOTA_DATE'].dt.strftime('%m')
# test_a2['day']=test_a2['QUOTA_DATE'].dt.strftime('%d')
# test_a2['QUOTA_NAME'].replace(['12190','12225','12260'],['最高温','最低温','平均温'],inplace=True)

# test_a2.info()
# test_a2 = pd.pivot_table(test_a2,index=['mon','QUOTA_NAME'],columns='day',values='QUOTA_VALUE')
# test_a2=test_a2.reset_index()
# test_a2=test_a2.values


list1 = [         
         ['00','00718','d'],
         ['00','11930','d'],    #公司总取水量  
         ['1001','04281','d'], #西村取水
         ['1001','00718','d'],
         ['1002','04281','d'],
         ['1002','00718','d'],
         ['1003','04281','d'],
         ['100301','04281','d'],
         ['100302','04281','d'],
         ['1003','00718','d'],
         ['100301','00718','d'],
         ['100302','00718','d'],
         ['1004','04281','d'],
         ['1004','00718','d'],
         ['1005','04281','d'],
         ['1005','00718','d'],
         ['1007','04281','d'],
         ['1007','00718','d'],
         ['100901','04281','d'],
         ['1016','04281','d'],
         ['1016','00718','d'],
         ]

shuju_df = tjfx.TjfxData().getdata(startd,endd,list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)
#shuju_df['QUOTA_NAME'].replace(['取水量','入厂取水量','水厂供水总量'],['0取水量','1入厂取水量','2供水总量'],inplace=True)
shuju_df['DEPT_QUOTA'] = shuju_df['QUOTA_NAME']+'_'+shuju_df['GROUP_NAME']


shuju_df['mon']=shuju_df['QUOTA_DATE'].dt.strftime('%m')
shuju_df['day']=shuju_df['QUOTA_DATE'].dt.strftime('%d')

test = pd.pivot_table(shuju_df,index=['mon','DEPT_QUOTA'],columns='day',values='QUOTA_VALUE')

test = test.unstack(level=0)
test.loc['水厂供水总量_江村水厂'] = test.loc['水厂供水总量_江村一厂'] + test.loc['水厂供水总量_江村二厂'] 
test.loc['入厂取水量_江村水厂'] = test.loc['入厂取水量_江村一厂'] + test.loc['入厂取水量_江村二厂'] 


test.loc['取水量_广州自来水公司'] = test.loc['入厂取水量_西江下陈'] + test.loc['入厂取水量_南洲水厂'] + test.loc['入厂取水量_新塘水厂'] + test.loc['入厂取水量_西洲水厂'] 

test.loc['水厂供水总量_广州自来水公司'] = test.loc['水厂供水总量_北部水厂'] + test.loc['水厂供水总量_南洲水厂'] + test.loc['水厂供水总量_新塘水厂'] + test.loc['水厂供水总量_石门水厂'] \
     + test.loc['水厂供水总量_西村水厂'] + test.loc['水厂供水总量_西洲水厂'] + test.loc['水厂供水总量_江村水厂']

test.drop(['水厂供水总量_江村一厂','水厂供水总量_江村二厂','入厂取水量_江村一厂','入厂取水量_江村二厂'],inplace=True)
test = test.stack(level=1, dropna=True) 
test = test.swaplevel('mon','DEPT_QUOTA', axis=0)

test1=test.reset_index()
test2=test1.values
test3 = np.concatenate((a,test2,test_a1,a2,a3), axis=0)

idex=np.lexsort([test3[:,1], test3[:,0]])
#先按第一列升序，再按第二列升序，.
#注意先按后边的关键词排序
sorted_data = test3[idex, :]
sorted_data = pd.DataFrame(sorted_data)
sorted_data.iloc[:,2:] = sorted_data.iloc[:,2:].apply((lambda x: pd.to_numeric(x,errors='coerce')))


sorted_data.columns = ['月份','指标'] + [str(i)+'日' for i in list(range(1,sorted_data.shape[1]-1))]
sorted_data = pd.melt(sorted_data,id_vars=['月份','指标'],var_name='日')
sorted_data['value'] = pd.to_numeric(sorted_data['value'],errors='coerce')

sorted_data = pd.pivot_table(sorted_data,index=['月份','日'],columns=['指标'],values='value',fill_value=0)
new_idx = sorted_data.index.map(lambda x: (endd[0:4]+'年'+x[0]+'月'+x[1]))
sorted_data.index = new_idx
sorted_data.index = pd.to_datetime(sorted_data.index,format='%Y年%m月%d日')
sorted_data.sort_index(inplace=True)
#a3[:,1]
#注意此处可能因为没有指标而出错，不会报错。
list1 = sorted_data.get("花都")


sorted_data = sorted_data.assign(开发区东4 = sorted_data.get('天河开发区路口DN600正向累计',0),
                   开发区东7 = sorted_data.get('姬堂村A表累计流量',0) + sorted_data.get('姬堂村B表累计流量',0),     
                 开发区东8 = sorted_data.get('科学城雕塑正向累计',0) + sorted_data.get('宏仁电子厂门口(0017988650)累计流量',0),
                 开发区东9 = sorted_data.get('开发区西区累计流量1',0) + sorted_data.get('开发区西区累计流量2',0) + sorted_data.get('萝岗开发区供水中心DN400累计',0) - sorted_data.get('萝岗开发区供水中心DN400反向累计',0) + sorted_data.get('开发区北面累计',0),
                 开发区东10 = sorted_data.get('新新大道与永安大道交界DN1200累计',0),
                 开发区北4 = sorted_data.get('天鹿北路累计',0) - sorted_data.get('天鹿北路反向累计',0),
                 花都北1 = sorted_data.get('太成村正向累计流量',0) + sorted_data.get('花都迎宾大道正向累计',0),
                 花都北2 = sorted_data.get('云山大道正向累计',0) + sorted_data.get('雅遥正向累计流量',0) + sorted_data.get('越堡水泥公司DN1400正向累计',0),
                 开发区北区= lambda x:(x['开发区北4']),
                 开发区东区=  lambda x:(sorted_data.get('开发区',0) - x['开发区北区']),
                 趸售 = sorted_data.get('开发区',0) + sorted_data.get('南9（黄岐）',0) + sorted_data.get('花都',0) + sorted_data.get('人和',0) + sorted_data.get('穗云',0),
                 黄岐 = sorted_data.get('南9（黄岐）',0),
                 公司取供比 = sorted_data.get('取水量_广州自来水公司',0) / sorted_data.get('水厂供水总量_广州自来水公司',0),
                 西村取供比 = sorted_data.get('入厂取水量_西村水厂',0) / sorted_data.get('水厂供水总量_西村水厂',0),
                 石门取供比 = sorted_data.get('入厂取水量_石门水厂',0) / sorted_data.get('水厂供水总量_石门水厂',0),
                 江村取供比 = sorted_data.get('入厂取水量_江村水厂',0) / sorted_data.get('水厂供水总量_江村水厂',0),
                 北部取供比 = sorted_data.get('入厂取水量_北部水厂',0) / sorted_data.get('水厂供水总量_北部水厂',0),
                 新塘取供比 = sorted_data.get('入厂取水量_新塘水厂',0) / sorted_data.get('水厂供水总量_新塘水厂',0),
                 西洲取供比 = sorted_data.get('入厂取水量_西洲水厂',0) / sorted_data.get('水厂供水总量_西洲水厂',0),
                 南洲取供比 = sorted_data.get('入厂取水量_南洲水厂',0) / sorted_data.get('水厂供水总量_南洲水厂',0),)


                 
                 

sorted_data = sorted_data.reindex(columns=['取水量_广州自来水公司', '入厂取水量_西村水厂','入厂取水量_石门水厂','入厂取水量_江村水厂','入厂取水量_北部水厂','入厂取水量_新塘水厂', '入厂取水量_西洲水厂', '入厂取水量_南洲水厂',
                                           '最高温','最低温','平均温',  
    '水厂供水总量_广州自来水公司','水厂供水总量_西村水厂','水厂供水总量_石门水厂','水厂供水总量_江村水厂', '水厂供水总量_北部水厂','水厂供水总量_新塘水厂', '水厂供水总量_西洲水厂', '水厂供水总量_南洲水厂', 
    '公司取供比','西村取供比','石门取供比','江村取供比','北部取供比','新塘取供比','西洲取供比','南洲取供比',
    '中区', '东区','南区', '北区', '趸售','开发区','黄岐','花都','人和', '穗云',
    '开发区东区','开发区东4','开发区东7','开发区东8','开发区东9','开发区东10','开发区北区', '开发区北4','花都北1','花都北2', 
    '中1（同德围）', '中2（同和）', '中3（罗冲围）','中4（二沙岛）', '中5（河沙）', '中6（中心城区）', 
    '东1(体育中心）', '东2（元岗）', '东3（龙洞）', '东4（大观）', '东5（东圃）', '东6（员村）','东7（文冲）', '东8（罗岗）', '东9（南岗）', '东10（新塘）', 
    '南1（新洲会展）', '南2（赤沙仑头北山）', '南3（土华小洲）', '南4（长洲岛）', '南5（大学城）', '南6（花地河东）', '南7（龙溪）', '南8（花地河西）', '南9（黄岐）','南10（金沙洲）', '南11（海珠城）',     
    '北1（机场）', '北2（江村东）', '北3（江村西）', '北4（嘉禾均禾）', '北5（石门石井）', '北6（机场路广园新村）','北7（白云新城）',     
    ])
 

sorted_data.index = sorted_data.index.map(lambda x:x.strftime('%Y年%m月%d日'))

sorted_data.to_excel(r'C:\Users\XieJie\mypyworks\输出\2021分区供水.xlsx')


