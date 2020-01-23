# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:06:18 2020

@author: XieJie
"""


import numpy as np
import matplotlib.pyplot as plt 
 

ax1 = plt.subplot(211) # 在图表2中创建子图1
ax2 = plt.subplot(212) # 在图表2中创建子图2
 
x = np.linspace(0, 3, 100)
i = 1
#for i in range(5):
plt.figure(1)  #选择图表1
plt.plot(x, np.exp(i*x/3))
plt.sca(ax1)   #选择图表2的子图1
plt.plot(x, np.sin(i*x))
plt.sca(ax2)  # 选择图表2的子图2
plt.plot(x, np.cos(i*x))

