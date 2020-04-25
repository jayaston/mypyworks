# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:33:12 2020

@author: Jay
"""

from bs4 import BeautifulSoup 
import requests
import pandas as pd
import datetime as dt
url='http://www.tianqihoubao.com/lishi/guangzhou/month/202001.html'
df = pd.read_html(url,encoding='gbk',header=0)[0]


df['最高温'] = df['气温'].str.split('/', 1).str[0].str.replace('℃','').str.strip().astype('float')
df['最低温'] = df['气温'].str.split('/', 1).str[1].str.replace('℃','').str.strip().astype('float')
df['日期'] = df['日期'].apply(lambda X : (dt.datetime.strptime(X,'%Y年%m月%d日')).date())
df = df[['日期','最高温','最低温']]
#headers = {
#    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
#}
#response = requests.get(url,headers=headers)
#soup = BeautifulSoup(response.text,'lxml')  #'lxml'解析器更好
## 找到表格整理并读取
#table = soup.find('table')
#df = pd.read_html(table.prettify(),header=0)[0]
dt.datetime.now().date()