# coding=utf-8
'''

读入并预处理数据
@author: Shuaichi Li
@email: shuaichi@mail.dlut.edu.cn
@date: 2020/09/13 11:02
'''

# import
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import random

class DataSet:
    def __init__(self,batch_size):
        current_path = os.getcwd()
        if current_path.endswith('src'):
            os.chdir(current_path.replace('src',''))
        self.data_path = './bin/data.txt'
        self.batch_size = batch_size
    
    '''读取数据集'''
    def get_dataset(self):
        dataset = []
        with open(self.data_path,'r',encoding='utf-8') as t:
            file = t.readlines()
            for line in file:
                line = line.strip(',\n')
                line = eval(line)
                dataset.append(line)
        x,y = list(zip(*dataset))
        return (np.array(x),np.array(y))

    '''绘制散点图'''
    def draw_scatter(self,x,y,x2=None,y2=None):
        plt.scatter(x, y, color='blue', marker='.')
        if x2 and y2:
            plt.plot(x2, y2, color='red', marker='.')
        plt.title('scatter chart')
        plt.show()

    '''绘制折线图'''
    def draw_line(self,x,y,x2=None,y2=None):
        plt.plot(x, y, color='blue', marker='.')
        if x2 and y2:
            plt.plot(x2, y2, color='red', marker='.')
        plt.title('line chart')
        plt.show()
    
    '''得到划分batch的数据'''
    def get_batch(self,x,y):
        data = list(zip(x,y))
        random.shuffle(data)
        data_len = len(data)
        if data_len%self.batch_size!=0:
            batch_num = data_len//self.batch_size + 1
        else:
            batch_num = data_len//self.batch_size
        batch_data = []
        for i in range(batch_num):
            if self.batch_size*(i+1)>data_len:
                one_batch = data[self.batch_size*i:]
            else:
                one_batch = data[self.batch_size*i:self.batch_size*(i+1)]
            batch_data.append(one_batch)
        return batch_data


