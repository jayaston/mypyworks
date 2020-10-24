# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:03:34 2020

@author: XieJie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

# 配置pandas显示
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

# 获取数据
dataset_train = pd.read_csv(r'c:\Users\XieJie\mypyworks\测试数据\NSE-TATAGLOBAL2.CSV')
# 多因素关联时，注意二维数组的形式
training_set = dataset_train.iloc[:, [1, 5, 6]].values #只取开盘价 收盘价 总交易量
real_stock_price = dataset_train.iloc[5:22, 1:2].values

# 归一化
sc_X = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc_X.fit_transform(training_set)

X_train = []
y_train = []

# 得到训练数据和标签
for i in range(5, 20):
    X_train.append(training_set_scaled[i-5:i, :])
    y_train.append(training_set_scaled[i:i+2, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

print(y_train[:10])
print(y_train.shape[1])

# 构造LSTM神经网络，Input_shape的格式和Dense的输出维数需要关注，多步预测
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(y_train.shape[1]))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 为加快测试，保存及加载模型文件
model.save(r"demo.model")


model = load_model(r"demo.model")
print(model.summary())

# 获取验证数据
X_test = []
for i in range(50, 2052):
    X_test.append(training_set_scaled[i-50:i, :])

X_test = np.array(X_test)
# 预测
predicted_stock_price = model.predict(X_test)

predicted_stock_price = sc_X.inverse_transform(predicted_stock_price)
print(predicted_stock_price[:-10])
print(real_stock_price[:-10])
predicted_stock_price = predicted_stock_price[:, 0]
print(len(real_stock_price), len(predicted_stock_price))

# 画图
plt.plot(real_stock_price[1500:], color='green', label='TATA Stock Price')
plt.plot(predicted_stock_price[1500:], color='red', label='Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()