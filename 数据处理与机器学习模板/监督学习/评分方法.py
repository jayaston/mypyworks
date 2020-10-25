# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 21:21:06 2020

@author: Jay
"""
pipline.score(X_test, Y_test) #缺省方法评分。
from sklearn.model_selection import cross_val_score 
scores = cross_val_score(dtc,X,Y,cv=5,scoring='f1') #用训练集进行交叉验证。模型评估效果更客观。
回归指标：
explained_variance_score(y_true,y_pred,sample_weight=None,multioutput=‘uniform_average’)#回归方差(反应自变量与因变量之间的相关程度)

mean_absolute_error(y_true,y_pred,sample_weight=None,multioutput=uniform_average’)#平均绝对误差

mean_squared_error(y_true, y_pred, sample_weight=None, multioutput=‘uniform_average’)#均方差

median_absolute_error(y_true, y_pred) #中值绝对误差

r2_score(y_true, y_pred,sample_weight=None,multioutput=‘uniform_average’) #：R平方值

#分类指标：
accuracy_score(y_true,y_pre) #准确率得分，是模型分类正确的数据除以样本总数 ，model.score(x_test,y_test)，效果一样。

auc(x, y, reorder=False) #ROC曲线下的面积;较大的AUC代表了较好的performance。

average_precision_score(y_true, y_score, average=‘macro’, sample_weight=None)#根据预测得分计算平均精度(AP)

brier_score_loss(y_true, y_prob, sample_weight=None, pos_label=None)#The smaller the Brier score, the better.

confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)#混淆矩阵，用来评估分类的准确性。有的分类问题，实际样本中1000个A，10个B，如果最后分类大多数B都被预测错误了，但依据其他评估方法，得分反而很高(因为A的数目相对太多导致的)，召回率和精确率的区别。

f1_score(y_true, y_pred, labels=None, pos_label=1, average=‘binary’, sample_weight=None)#F1值　　F1 = 2 * (precision * recall) / (precision + recall) precision(查准率)=TP/(TP+FP) recall(查全率)=TP/(TP+FN)

log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)#对数损耗，又称逻辑损耗或交叉熵损耗

precision_score(y_true, y_pred, labels=None, pos_label=1, average=‘binary’,) #查准率或者精度； precision(查准率)=TP/(TP+FP)

recall_score(y_true, y_pred, labels=None, pos_label=1, average=‘binary’, sample_weight=None)#查全率 ；recall(查全率)=TP/(TP+FN)

roc_auc_score(y_true, y_score, average=‘macro’, sample_weight=None)#计算ROC曲线下的面积就是AUC的值，the larger the better

roc_curve(y_true,y_score,pos_label=None,sample_weight=None,drop_intermediate=True)#计算ROC曲线的横纵坐标值，TPR，FPR TPR = TP/(TP+FN) = recall(真正例率，敏感度) FPR = FP/(FP+TN)(假正例率，1-特异性)

classification_report(y_test,y_log_pre)#其中的各项得分的avg/total 是每一分类占总数的比例加权算出来的
print(classification_report(y_test,y_log_pre))
