# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:48:26 2020

@author: Jay
"""

import requests
from  bs4 import BeautifulSoup
import pandas as pd
url = 'http://www.weather.com.cn/weather1d/101280101.shtml'
html = requests.get(url)
html.encoding = 'utf-8'
soup = BeautifulSoup(html.text,'html.parser',from_encoding='utf-8')
res_data = soup.findAll('script') #获取页面内的所有<script>标签
weather_data = res_data[3]        #获取第5个<script>标签，返回一个list
for x in weather_data:
   weather1 = x           #因为weather_data是一个list，我们取list的第一个

index_start = weather1.find("{")  #目前weather1还是一个字符串，需要将里面的json截取出来
index_end = weather1.find(";")
weather_str = weather1[index_start:index_end]
weather = eval(weather_str)        #将字符串转换成字典

weather_dict = weather["od"] 
weather_date = weather_dict["od0"]   #时间
weather_position_name = weather_dict["od1"] #地点
weather_list = list(reversed(weather["od"]["od2"]))
insert_list = []                     #存放每小时的数据的list，用于之后插入数据库
for item in weather_list:
  #od21小时，od22温度，od26降雨，od24风向，od25风力
  weather_item = {}
  weather_item['时间'] = item['od21']
  weather_item['温度'] = item['od22']
  weather_item['降雨量'] = item['od26']
  weather_item['湿度'] = item['od27']
  weather_item['风向'] = item['od24']
  weather_item['风力'] = item['od25']
  weather_item['od23'] = item['od23']
  insert_list.append(weather_item)

#打印查看变量
weather_df = pd.DataFrame(insert_list)