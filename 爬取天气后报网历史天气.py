# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:33:12 2020

@author: Jay
"""
# from bs4 import BeautifulSoup 
# import requests
import pandas as pd
import datetime as dt
import time
months = pd.Series(pd.date_range('2015-01-01','2020-04-30',freq='MS')).dt.strftime('%Y%m')
weather_df = pd.DataFrame()
for month in months:
    print('正在获取'+month+'天气...')
    url='http://www.tianqihoubao.com/lishi/guangzhou/month/'+month+'.html'
    while True :
        try:            
            df = pd.read_html(url,encoding='gbk',header=0)[0]
            weather_df = weather_df.append(df)
            break
        except:
            time.sleep(60)
    
    
weather_df['最高温'] = weather_df['气温'].str.split('/', 1).str[0].str.replace('℃','').str.strip().astype('float')
weather_df['最低温'] = weather_df['气温'].str.split('/', 1).str[1].str.replace('℃','').str.strip().astype('float')
weather_df['日期'] = weather_df['日期'].apply(lambda X : (dt.datetime.strptime(X,'%Y年%m月%d日')).date())
weather_df = weather_df[['日期','最高温','最低温']]

weather_df.to_csv(r'~/mypyworks/自来水数据/2015-2020气温数据.csv')


#headers = {
#    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
#}
#response = requests.get(url,headers=headers)
#soup = BeautifulSoup(response.text,'lxml')  #'lxml'解析器更好
## 找到表格整理并读取
#table = soup.find('table')
#df = pd.read_html(table.prettify(),header=0)[0]
