# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:26:39 2020

@author: XieJie
"""

 
    
import urllib.request,urllib.parse, sys
import ssl
host = 'https://aliv8.data.moji.com'
path = '/whapi/json/aliweather/forecast15days'
method = 'POST'
appcode = '43fca530b164413d82b5af24000a56e5'
querys = ''
bodys = {}
url = host + path

bodys['lat'] = '39.91488908'
bodys['lon'] = '116.40387397'
bodys['token'] = '7538f7246218bdbf795b329ab09cc524'
post_data = bytes(urllib.parse.urlencode(bodys), encoding='utf8')
request = urllib.request.Request(url)
request.add_header('Authorization', 'APPCODE ' + appcode)
#根据API的要求，定义相对应的Content-Type
request.add_header('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
response = urllib.request.urlopen(request, data = post_data)
content = response.read()
if (content):
    print(content)