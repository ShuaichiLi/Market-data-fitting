# coding=utf-8
'''
计算R^2 和 相关系数
@author: Shuaichi Li
@email: shuaichi@mail.dlut.edu.cn
@date: 2020/09/14 10:44
'''

# import
import numpy as np
import math 

'''计算R^2'''
def cal_R2(y_pred,y_train):
    eps = 1e-9
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)
    y_mean = sum(y_train)/len(y_train)

    up   = sum((y_pred-y_train)**2)
    down = sum((y_mean-y_train)**2)
    
    print('R^2：',1-up/(down+eps))

'''计算r'''
def cal_r(y_pred,y_train):
    eps = 1e-9
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)
    y_pred_mean =  sum(y_pred)/len(y_pred)
    y_train_mean = sum(y_train)/len(y_train)

    up   = sum((y_pred-y_pred_mean)*(y_train-y_train_mean))
    down = math.sqrt(sum((y_pred-y_pred_mean)**2)*sum((y_train-y_train_mean)**2))

    print('相关系数r：',up/(down+eps))
