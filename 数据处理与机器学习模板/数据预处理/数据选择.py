# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:39:28 2020

@author: Jay
"""
# read_table（sep=’,’,enconding=’utf-8’）、read_csv(sep=’,’,enconding=’utf-8’)
#pd.read_excel（io，sheet_name = 0，header = 0，names = None或者要使用的列名列表，
#index_col = None，usecols = int或list（支持切片元素），
#默认为None，squeeze = 默认为False,如果解析的数据只包含一列，则返回一个Series。,
#dtype = None或者{'a'：np.float64，'b'：np.int32}, ...）

df_2 = pd.read_csv("sales_data_types.csv",dtype={"Customer_Number":"int"},converters={
    "2016":convert_currency,
    "2017":convert_currency,
    "Percent Growth":convert_percent,
    "Jan Units":lambda x:pd.to_numeric(x,errors="coerce"),
    "Active":lambda x: np.where(x=="Y",True,False)
})

#处理日期型数据
def parse(y,m,d,h):
    return dt.datetime.strptime(' '.join([y,m,d,h]), '%Y %m %d %H')

dataset = pd.read_csv(r'.\mypyworks\StatLedger\数据表\raw.csv', 
                      parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)




#划分数据集
train = sup_water['2020-1-27':'2020-2-13']#训练数据集
validation = sup_water['2020-2-14':'2020-2-16']#验证数据集，用于搜索最佳参数。
test = sup_water['2020-2-17':'2020-2-19']#测试数据集。

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#按照某个分类变量每一类的样本20%的比例来划分训练和测试集。
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
     
    
#填补缺失值
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)
X = imputer.transform(housing_num)


#独热编码
from sklearn.preprocessing import OneHotEncoder
preprocessing.OneHotEncoder().transform(array)#返回稀疏矩阵 .toarray() 转化为数组
cat_encoder = OneHotEncoder(sparse=False)#不转化稀疏矩阵
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
#顺序分类编码
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
#标准化
from sklearn.preprocessing import StandardScaler
preprocessing.StandardScaler()#标准化转化为Z分位数？
preprocessing.MinMaxScaler()#规范在0-1之间
MaxAbsScaler（）#规范在-1到1之间
sklearn.preprocessing.Normalizer(norm=’l2’, copy=True)#正则化 每行的元素除以该行的范数
#||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p,l1的P=1，l2的p=2
preprocessing.Binarizer()#特征的二值化是指将数值型的特征数据转换成布尔类型的值。默认是根据0来二值化，大于0的都标记为1，小于等于0的都标记为0，参数threshold 来设置阀值




