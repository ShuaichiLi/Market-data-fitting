# coding=utf-8
'''
main函数
@author: Shuaichi Li
@email: shuaichi@mail.dlut.edu.cn
@date: 2020/09/14 10:38
'''

# import
from linear_reg import linear_reg,train,test
from dataset import DataSet
import numpy as np
from utils import cal_r,cal_R2




if __name__ == '__main__':
    '''模型路径'''
    model_path = './bin/model_params.bin'

    '''读取数据集'''
    s = DataSet(batch_size = 5)
    x,y = s.get_dataset()
    
    '''绘制散点图和曲线图'''
    s.draw_scatter(list(x),list(y))
    s.draw_line(list(x),list(y))


    '''定义模型'''
    model = linear_reg(learning_rate = 0.1,degree = 7)
    '''训练模型'''
    # train(model,x,y,s.get_batch(x,y),epoch = 1000)
    '''保存模型'''
    # model.save_model(model_path)
    '''读取模型'''
    model.load_model(model_path)
    '''预测数据'''
    new_y = test(model,x)
    '''绘制曲线'''
    s.draw_line(x,new_y,list(x),list(y))
    '''计算评价指标'''
    cal_R2(new_y,y)
    cal_r(new_y,y)
    
