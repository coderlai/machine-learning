import numpy as np
# sigmod 激活函数
def sigmoid(x):
	return 1/(1+np.exp(-x))
	
# softmax输出函数
def softmax(x):
	c = np.max(x)
	exp_a = np.exp(x-c)  #防止溢出
	sum_exp_a = np.sum(exp_a)
	y = exp_a/sum_exp_a
	return y
	
	