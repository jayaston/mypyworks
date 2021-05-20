# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:48:26 2020

@author: Jay
"""
import re
import requests
from  bs4 import BeautifulSoup
import pandas as pd

from bs4 import UnicodeDammit
import urllib.request


url = 'http://www.weather.com.cn/weather40d/101280101.shtml'
url1= 'http://www.weather.com.cn/weather/101280101.shtml'
url2='http://www.weather.com.cn/weather15d/101280101.shtml'
url3= 'https://www.tianqi.com/guangzhou/40/'
myheader = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36',
           'Accept': 'application/json, text/javascript, */*; q=0.01'}

myheader1 = {"User-Agent":"Mozilla/5.0(Windows; U; Windows NT 6.0 x64; en-US; rv:1.9pre)Gecko/2008072421 Minefield/3.0.2pre"
        }

myheader2= {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"
    }

req = urllib.request.Request(url1, headers=myheader1)
data = urllib.request.urlopen(req)
data = data.read()
dammit = UnicodeDammit(data, ["utf-8","gbk"])
data = dammit.unicode_markup
soup = BeautifulSoup(data, features="html.parser")
lis = soup.select("td")


response = requests.get(url, headers=myheader2)
response = response.text.encode("raw_unicode_escape").decode("utf-8")
soup = BeautifulSoup(response,"html5lib")
div_tatall = soup.find("span",class_="nowday") #find() 找符合要求的第一个元素





myheader3 = {"User-Agent": "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;","Accept": "application/json, text/javascript, */*; q=0.01"
    }

html = requests.get(url,headers=myheader3)
html.encoding = 'utf-8'
soup = BeautifulSoup(html.text,"html.parser",from_encoding='utf-8')
soup.find('span',class_='w_day')
soup.find_all(text=re.compile("35℃"))
soup.select("tbody tr")[-1].select("td")[-1].select("div[class='ks'] ")[0].find('span')
res_data = soup.findAll("ul[class='t clearfix'] li") 
res_data = soup.select("td") #获取页面内的所有<script>标签
weather_data = res_data[10]        #获取第5个<script>标签，返回一个list
for x in weather_data:
   weather1 = x           #因为weather_data是一个list，我们取list的第一个



text = html.text
text.find("体验新版")

text[-1]


weather1[index_start:index_end]


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