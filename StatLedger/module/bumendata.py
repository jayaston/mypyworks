# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:58:56 2019

@author: XieJie
"""
import pandas as pd
import pymysql
import traceback

class BumenData:
    def __init__(self):
        host='10.1.80.197'
        self.port=3306
        self.database='TJFX'
        self._user='admin'
        self._password='200925'
        self.charset='utf8'
        self.conn = pymysql.connect(host=host,port=self.port,database=self.database,
                               user=self._user,password=self._password,charset=self.charset)
    def getdata(self,startd,endd,quotas=None):        
        try:
            sql = "select QUOTA_DATE,QUOTA_DEPT_CODE ,QUOTA_CODE,QUOTA_VALUE,RECORD_TYPE \
            from tj_data \
            where date_format(quota_date,'%Y%m%d') >= '"+startd+"' \
            and date_format(quota_date,'%Y%m%d') <='"+endd+"'"
            df = pd.read_sql(sql,self.conn)
            df_result = pd.DataFrame()       
            for item in quotas:                        
                df_result_new =df[(df.QUOTA_DEPT_CODE==item[0]) & (df.QUOTA_CODE==item[1]) & (df.RECORD_TYPE==item[2])]
                df_result = pd.concat([df_result,df_result_new])
        except Exception as e:
            print("%s;您没有指定具体指标，返回全部数据"%e)
            df_result = df        
        print(df_result.head())
        self.conn.close()
        return(df_result)
        
        
    def importdata(self,mylist:list):
        c = self.conn.cursor()  # 使用 cursor() 方法创建一个游标对象 cursor
        try:
            sql = "drop table TJ_DATA_TMP3"
            c.execute(sql)#删除已经存在的临时表
            self.conn.commit()#提交到数据库执行
        except:            
            self.conn.rollback()#发生错误时，回滚！
            
        sql="CREATE TABLE TJ_DATA_TMP3 ( \
            QUOTA_CODE VARCHAR ( 12 ), \
            MON VARCHAR ( 6 ), \
            QUOTA_DATE VARCHAR ( 20 ), \
            QUOTA_VALUE VARCHAR ( 64 ), \
            REPORT_FLAG CHAR ( 1 ), \
            QUOTA_DEPT_CODE VARCHAR ( 36 ), \
            IMPORT_FLOW_NO VARCHAR ( 12 ), \
            WARNING_CODE VARCHAR ( 12 ), \
            RECORD_TYPE CHAR ( 1 ), \
            CONSTRAINT pk_tmp3 PRIMARY KEY ( QUOTA_CODE, QUOTA_DATE, QUOTA_DEPT_CODE, RECORD_TYPE ))"
            
        try:
            c.execute(sql)       #删除已经存在的临时表
            self.conn.commit()#提交到数据库执行
        except:            
            print("导入失败，建立零时表错误！")
            traceback.print_exc()
            self.conn.rollback()#发生错误时，回滚！        
#        try:
#            c.execute('truncate table TJ_DATA_SJY')#清除临时表数据
#            self.conn.commit()#提交到数据库执行
#        except:            
#            self.conn.rollback()#发生错误时，回滚！
        else:
            sql = "INSERT INTO TJ_DATA_TMP3 \
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    
            try:
                c.executemany(sql,mylist)#执行sql语句
                self.conn.commit()#提交到数据库执行
            except:
                print ('导入失败，报表数据导入错误！')
                traceback.print_exc()
                self.conn.rollback()#发生错误时，回滚！
            else:
                sql = "INSERT INTO tj_data ( QUOTA_DATE, QUOTA_DEPT_CODE, QUOTA_CODE, QUOTA_VALUE, RECORD_TYPE ) SELECT \
                str_to_date(QUOTA_DATE,'%Y-%m-%d %H:%i:%s') as QUOTA_DATE, \
                QUOTA_DEPT_CODE, \
                QUOTA_CODE, \
                QUOTA_VALUE, \
                RECORD_TYPE  \
                FROM \
                	TJ_DATA_TMP3 t \
                	ON DUPLICATE KEY UPDATE QUOTA_VALUE = t.QUOTA_VALUE"
                try:
                    c.execute(sql)#执行sql语句
                    self.conn.commit()#提交到数据库执行
                    print('数据上传完毕！')   
                except:
                    print ('导入失败，写入主数据错误')
                    traceback.print_exc()
                    self.conn.rollback()#发生错误时，回滚！                   
 #    c.executemany(sql, dir_data(r'E:\pyworks\行业表'))  # 执行sql语句             
      
        c.close()  # 关闭连接
        self.conn.close()
        
        
if __name__ == "__main__":
    b = BumenData()
    df2 = b.getdata('20181231', '20190101')