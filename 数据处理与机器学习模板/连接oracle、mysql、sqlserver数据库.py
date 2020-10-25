# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import cx_Oracle
conn = cx_Oracle.connect('')
print (conn.version)  
c=conn.cursor()
c.execute("select QUOTA_DATE,QUOTA_DEPT_CODE ,QUOTA_CODE,QUOTA_VALUE,RECORD_TYPE \
          from zls_tjfx.tj_quota_data \
          where to_char(quota_date,'yyyymmdd') >= '20190101' \
          and to_char(quota_date,'yyyymmdd') <= '20190128' \
          and QUOTA_DEPT_CODE='1009' \
          and QUOTA_CODE='11930'")
rs = c.fetchall()

print ( rs[0,3]).decode('utf-8')

c.execute("insert into test_ccc values(1,sysdate,'我们')")
conn.commit()


sql1 = "select QUOTA_DATE,QUOTA_DEPT_CODE ,QUOTA_CODE,QUOTA_VALUE,RECORD_TYPE \
       from zls_tjfx.tj_quota_data \
       where to_char(quota_date,'yyyymmdd') >= '20190101' \
       and to_char(quota_date,'yyyymmdd') <= '20190128' \
       and QUOTA_DEPT_CODE='1009' \
       and QUOTA_CODE='11930'"
       
df = pd.read_sql(sql1,conn)     
df.head()         

conn.close()


import pymysql
conn2 = pymysql.connect(
        host='',
        port=3306,
        database='',
        user='',
        password='',
        charset='utf8')

sql2 = "select QUOTA_DATE,QUOTA_DEPT_CODE ,QUOTA_CODE,QUOTA_VALUE,RECORD_TYPE \
from tj_data \
where date_format(quota_date,'%Y%m%d') >= '20190101' \
and date_format(quota_date,'%Y%m%d') <='20190129' \
and quota_code = '31200'"

df2 = pd.read_sql(sql2,conn2)
conn2.close()

import pymssql
conn3 = pymssql.connect(host='',
                        port='1433',
                        database='ReadAliWeather',
                        user='',
                        password='')
sql3 = "select *  from \
AliWeather2019 \
where CONVERT(varchar(12) , fDate, 112 ) >= '20190101' \
and CONVERT(varchar(12) , fDate, 112 ) <= '20190131' \
and fNo in (1,4)"

df3 = pd.read_sql(sql3,conn3) 
conn3.close()
                                       


[(i,) for i in range(100)]