# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:50:53 2019

@author: XieJie
"""
import pandas as pd
import cx_Oracle
import readConfig
#from sqlalchemy import create_engine
#from sqlalchemy.types import CHAR,VARCHAR,DateTime,Numeric,Integer

class TjfxData:    
    def __init__(self,
                 user=readConfig.tjfx_user,
                 password=readConfig.tjfx_pwd,
                 dsn=readConfig.tjfx_dsn):
        self.conn = cx_Oracle.connect(user=user,password=password,dsn=dsn) 
        #conn_string='oracle+cx_oracle://'+self.user+':'+self.password+'@'+self.dsn       
    def getdata(self,startd,endd,quotas=None):
        
        sql1 = "select QUOTA_DATE,QUOTA_DEPT_CODE ,QUOTA_CODE,QUOTA_VALUE,RECORD_TYPE \
            from zls_tjfx.tj_quota_data \
            where quota_date >= to_date('"+startd+"','yyyymmdd') \
            and quota_date <= to_date('"+endd+"','yyyymmdd') \
            and to_char(quota_date,'ss')='00'"      
            
        sql2 = "SELECT DISTINCT QUOTA_CODE, QUOTA_NAME \
            FROM zls_tjfx.cs_Quota_Define" 
        sql3 = "SELECT DISTINCT GROUP_CODE , GROUP_NAME \
            FROM zls_tjfx.hr_organization"         
        
        try:           
            df = pd.read_sql(sql1,self.conn)
            df_result = pd.DataFrame()        
            for item in quotas:                        
                df_result_new =df[(df.QUOTA_DEPT_CODE==item[0]) & (df.QUOTA_CODE==item[1]) & (df.RECORD_TYPE==item[2])]
                df_result = pd.concat([df_result,df_result_new])
        except Exception as e:
            print("%s;可能因为您没有指定具体指标，将返回全部数据"%e)
            df_result = df        
        df_zhibiao = pd.read_sql(sql2,self.conn)
        df_bumen = pd.read_sql(sql3,self.conn)
        df_result = pd.merge(df_result,df_zhibiao,on='QUOTA_CODE')
        df_result = pd.merge(df_result,df_bumen,left_on='QUOTA_DEPT_CODE',right_on='GROUP_CODE').drop('GROUP_CODE',axis=1)
        self.conn.close()
        return df_result
    
    def get_all_quota(self):
        sql = "SELECT QUOTA_CODE,QUOTA_NAME \
            FROM zls_tjfx.cs_Quota_Define \
                WHERE EFFECTYPE = 'Y'"
        df_zhibiao = pd.read_sql(sql,self.conn) 
        self.conn.close() 
        result = df_zhibiao.drop_duplicates(['QUOTA_CODE'])
        return result
    
    def get_all_dept(self):
        sql = "SELECT DISTINCT GROUP_CODE , GROUP_NAME \
            FROM zls_tjfx.hr_organization"  
        df_bumen = pd.read_sql(sql,self.conn) 
        self.conn.close()
        result = df_bumen.drop_duplicates(['GROUP_CODE','GROUP_NAME'])
        return result
    def get_formula(self):
        sql = "SELECT i.TZ_TYPE ,r.QUOTA_CODE, q.QUOTA_NAME,r.ZB_DEPT_CODE,o.GROUP_NAME,r.FORMULA_CODE,f.FORMULA,f.START_TIME,f.END_TIME,v.VIEW_NAME 方案,i.TZ_NAME as 目录 \
            FROM CS_TZZB_RELATION r,CS_TZ_VIEW v, CS_TZ_ITEM i ,cs_formula_set f,CS_QUOTA_DEFINE q, HR_ORGANIZATION o \
                WHERE r.TZ_CODE = i.TZ_CODE(+)\
                    AND r.VIEW_CODE = v.VIEW_CODE(+)\
                        AND r.FORMULA_CODE = f.FORMULA_CODE(+)\
                            AND r.QUOTA_CODE = q.QUOTA_CODE(+)\
                                AND r.ZB_DEPT_CODE = o.GROUP_CODE(+)\
                                    AND r.FORMULA_CODE is not null \
                                        AND v.VIEW_NAME IN ('新系统')"  
        df_formula = pd.read_sql(sql,self.conn) 
        self.conn.close()
        result = df_formula.drop_duplicates(['TZ_TYPE','QUOTA_CODE','ZB_DEPT_CODE','FORMULA_CODE'])
        return result
    def get_formula_detail(self):
        sql = "select g.flow_no,g.operation,g.left_bracket,g.parameter,g.QUOTA_DEPT_CODE,g.quota_code,g.right_bracket,g.FORMULA_CODE\
            from zls_tjfx.Cs_Formula_Detail g"  
        df_formula = pd.read_sql(sql,self.conn) 
        self.conn.close()
        result = df_formula
        return result
        
    def importdata(self,mylist:list):
        c = self.conn.cursor()  # 使用 cursor() 方法创建一个游标对象 cursor
        try:
            sql = "drop table TJ_DATA_"+"tmp3"
            c.execute(sql)#删除已经存在的临时表            
        except :
            pass
        sql="CREATE TABLE TJ_DATA_"+"tmp3"+" ( \
            QUOTA_CODE VARCHAR2 ( 12 ), \
            MON VARCHAR2 ( 6 ), \
            QUOTA_DATE VARCHAR2 ( 20 ), \
            QUOTA_VALUE VARCHAR2 ( 64 ), \
            REPORT_FLAG CHAR ( 1 ), \
            QUOTA_DEPT_CODE VARCHAR2 ( 36 ), \
            IMPORT_FLOW_NO VARCHAR2 ( 12 ), \
            WARNING_CODE VARCHAR2 ( 12 ), \
            RECORD_TYPE CHAR ( 1 ), \
            CONSTRAINT pk_"+"tmp3"+" PRIMARY KEY ( QUOTA_CODE, QUOTA_DATE, QUOTA_DEPT_CODE, RECORD_TYPE ))"
        try:            
            c.execute(sql)       #创建新的临时表        
        except Exception as e: 
            print("无法创建临时表！导入失败")
            print(e)
        else:
            #try:
            #c.execute('truncate table TJ_DATA_SJY')#清除临时表数据
            #except:    
            sql = "INSERT INTO TJ_DATA_"+"tmp3"+" \
                VALUES(:1,:2,:3,:4,:5,:6,:7,:8,:9)"
    
            try:
                c.executemany(sql,mylist)#执行sql语句
                self.conn.commit()#提交到数据库执行
            except Exception as e:
                print('报表数据写入数据库错误！导入失败')
                print(e)
                self.conn.rollback()#发生错误时，回滚！
            else:
                sql = "MERGE INTO ZLS_TJFX.TJ_QUOTA_DATA z USING \
                (SELECT to_date ( QUOTA_DATE, 'yyyy-mm-dd hh24:mi:ss' ) AS QUOTA_DATE, \
                to_number(MON) AS MON, \
                QUOTA_DEPT_CODE, \
                QUOTA_CODE, \
                QUOTA_VALUE, \
                RECORD_TYPE \
                FROM \
                ZLS_TJFX.TJ_DATA_"+"tmp3"+" ) t ON ( \
                z.QUOTA_DATE = t.QUOTA_DATE \
                AND z.QUOTA_CODE = t.QUOTA_CODE \
                AND z.QUOTA_DEPT_CODE = t.QUOTA_DEPT_CODE \
                AND z.RECORD_TYPE = t.RECORD_TYPE  ) \
                WHEN matched THEN \
                UPDATE \
                SET z.QUOTA_VALUE = t.QUOTA_VALUE \
                WHEN NOT matched THEN \
                INSERT ( z.quota_date, z.mon,z.quota_dept_code, z.quota_code, z.quota_value, z.record_type ) \
                VALUES \
                ( t.quota_date, t.mon,t.quota_dept_code, t.quota_code, t.quota_value, t.record_type )"           
            
                try:
                    c.execute(sql)#执行sql语句
                    self.conn.commit()               
                except Exception as e:
                    print ('写入台账主数据错误！导入失败')
                    print(e)
                    self.conn.rollback()#发生错误时，回滚！ 
                else:
                     print("数据导入成功！")#提交到数据库执行
                #    c.executemany(sql, dir_data(r'E:\pyworks\行业表'))  # 执行sql语句  
                       
        c.close()  # 关闭游标      
        self.conn.close()#关闭连接
    def imp_df(self,df):  #数据框写入统计台账
        #df可能需要字符型
        df = df.reindex(columns=['QUOTA_CODE','MON','QUOTA_DATE','QUOTA_VALUE',
                             'REPORT_FLAG','QUOTA_DEPT_CODE','IMPORT_FLOW_NO',
                             'WARNING_CODE','RECORD_TYPE'])
        df['MON'] = df['QUOTA_DATE'].dt.strftime('%Y%m')
        df['QUOTA_DATE'] = df['QUOTA_DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['QUOTA_VALUE'] = df['QUOTA_VALUE'].astype('str')
        mylist = list(df.to_records(index=False))        
        self.importdata(mylist)
    def exp_formula(self):
        pass 
    
    def get_any_data(self,sql:str):
        df = pd.read_sql(sql,self.conn) 
        self.conn.close()#关闭连接
        return df
    
    def close(self):
        self.conn.close()
    
 
if __name__=="__main__":
    tz = TjfxData()
#    quotas = [['00','00752','m']
#             ]
#    df1 = a.getdata('20181231','20190101')
    df2 = tz.get_all_dept()
    df3 = tz.get_any_data("select * \
                          form zls_tjfx.tj_quota_data \
                          where quota_date = to_date('20200101','%Y%m%d')\
                          and record_type = 'm'")
#    df2.to_excel(r'C:\\Users\\XieJie\\mypyworks\\部门表.xlsx')
    from sklearn import datasets
    import pandas as pd
    datasets_iris = datasets.load_iris()
    iris = pd.DataFrame(datasets_iris.data,columns=datasets_iris.feature_names)
    test = list(iris.to_records(index=False))
