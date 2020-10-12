# coding=utf-8
'''
线性回归
@author: Shuaichi Li
@email: shuaichi@mail.dlut.edu.cn
@date: 2020/09/13 17:04
'''

# import
import random
random.seed(10)


class linear_reg:
    def __init__(self,learning_rate,degree):
        self.params = [0]*degree
        for i in range(len(self.params)):
            self.params[i] = random.random()
        self.learning_rate = learning_rate
    '''前向传播'''
    def forward(self,x):
        y = 0
        for i, w in enumerate(self.params):
            y += (x**i) * w
        return y
    '''平方误差'''
    def squared_loss(self,y_pred,y):
        return (y_pred-y)**2/2
    '''反向传播'''
    def backward(self,x,y_pred,y):
        delta = y_pred-y
        for i,_ in enumerate(self.params):
            self.params[i] = self.params[i]-self.learning_rate*(x**i)*delta
    '''保存模型'''
    def save_model(self,path):
        with open(path,'w') as t:
            t.write(str(self.params))
    '''读取模型'''
    def load_model(self,path):
        with open(path,'r') as t:
            self.params = eval(t.read().strip())
'''训练模型'''
def train(model,x,y,batch_data,epoch):
    for _ in range(epoch):
        for one_batch in batch_data:
            loss_batch = 0
            for x_train,y_train in one_batch:
                y_pred = model.forward(x_train)
                loss = model.squared_loss(y_pred,y_train)
                loss_batch+=loss
            model.backward(x_train,y_pred,y_train)
    print(model.params)
'''使用模型预测'''
def test(model,x):
    new_y = []
    for x_in in x:
        new_y.append(model.forward(x_in))
    return new_y