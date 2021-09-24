﻿# -*- coding: utf-8 -*-
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
startd = '20210901'
endd =   '20210921' #不能跨年
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
    by.x = 'QUOTA_DEPT_CODE', by.y = 'fmain',all.y=TRUE)

fenqushuiliang%>%dcast(fMonth+fName~fDay,value.var="QUOTA_VALUE",fill = '')->fenqushuiliang

fenqushuiliang$fName <- iconv(as.vector(fenqushuiliang$fName), 'GB2312', 'UTF-8')
close(channel2)
'''

robjects.r(r_script)
matrix = robjects.r['fenqushuiliang']#将dataframe变量通过运行变量名调出来
matrix1 = robjects.r['tianqiqingkuang.qiwen']
#耗水30976，稽查24299，每月发单23765，每月追收24293，每月剔除24294，两月发单23594，两月追收23595，两月剔除23596
a = np.array(matrix)#转化为二维数组
a = a.transpose()

a1 = np.array(matrix1)#转化为二维数组
a1 = a1.transpose()

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

list1 = [         
         ['00','00718','d'],
         ['00','11930','d']    #公司总取水量        
         ]
shuju_df = tjfx.TjfxData().getdata(startd,endd,list1)

shuju_df.info()

shuju_df.QUOTA_VALUE = pd.to_numeric(shuju_df.QUOTA_VALUE,errors='coerce').fillna(0)

shuju_df['mon']=shuju_df['QUOTA_DATE'].dt.strftime('%m')
shuju_df['day']=shuju_df['QUOTA_DATE'].dt.strftime('%d')

test = pd.pivot_table(shuju_df,index=['mon','QUOTA_NAME'],columns='day',values='QUOTA_VALUE')
test1=test.reset_index()
test2=test1.values
test3 = np.concatenate((a,test2,test_a1), axis=0)

idex=np.lexsort([test3[:,1], test3[:,0]])
#先按第一列升序，再按第二列升序，.
#注意先按后边的关键词排序
sorted_data = test3[idex, :]
sorted_data = pd.DataFrame(sorted_data)

sorted_data.to_excel(r'C:\Users\XieJie\mypyworks\输出\2021分区供水.xlsx')
