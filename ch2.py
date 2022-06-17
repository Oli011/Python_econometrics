#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 23:03:38 2022

@author: oli
"""

import pandas as pd
import wooldridge as woo
import numpy as np
import statsmodels.api as sm

############# C1 ################

data = woo.dataWoo('401K')

# i 求prate和mrate平均值
prate_average = data['prate'].mean()
mrate_average = data['mrate'].mean()

# ii OLS Regression Results 
y = data['prate']
x = data['mrate']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model_summary = model.summary()
print(model_summary)
# 样本容量 N（No. Observations）：1534
# R-squared:0.075

# iii 解释方程中的截距和mrate的系数

# 截距b0（const：83.0755）Intercept = 83.0755
# 系数b1（mrate：5.8611）
# 截距可以被视为 401K 参与的“基本”水平，
# 没有匹配的贡献（83% 的劳动力不需要诱因）。 
# mrate 的系数是匹配率的每一美元的参与增加
#（也就是说，对于匹配率的每一美元，我们获得略低于 6% 的额外参与）

# iv mrate = 3.5，求prate的预测

X_p = pd.DataFrame({"const" : [1.0], "mrate" : [3.5]})

prate_pre = model.predict(exog=X_p)

# 103.589233。这一预测意味着，
# 一旦雇主贡献 3.5 倍或更多，将有超过所有符合条件的参与者参与。
# 该结果反映了线性模型的斜率以恒定速率增加，而不是任何观察到的结果。

# v prate波动，有多少是可以由mrate解释
# 总的 = 可解释的部分 + 不可解释的部分（残差）
# 可解释的部分：R-squared:0.075 
# R2 越大越好，可解释性越强
# 参与率的 7.5% 的变化是由匹配率来解释的。
# 这不是一个非常大的数量，仅由匹配率来解释。

############# C2 ################

data = woo.dataWoo('CEOSAL2')

# i 求salary和ceoten平均值
salary_average = data['salary'].mean()
ceoten_average = data['ceoten'].mean()

# ii 多少 ceoten = 0 最长的
ceoten_0 = data[data['ceoten'] == 0]
ceoten_0_num = ceoten_0.shape[0]
ceoten_max = data['ceoten'].max()

# iii 估计简单回归模型
y = data['lsalary']
x = data['ceoten']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model_summary = model.summary()
print(model_summary)

# N = 177, Intercept: 6.5055, 
# ceoten = 0.0097, R-Squared 0.013.
# 系数：如果再担任 CEO 一年，工资的大致增长百分比约为 1%。

# ############# C3 ################

data = woo.dataWoo('SLEEP75')

# (1) 
y = data['sleep']
x = data['totwrk']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model_summary = model.summary()
print(model_summary)

# 样本量：706
# R2 = 10.3%
# 截距：3586.3770  基本水平

# （2） 增加两小时
# 系数：-0.1507   2小时——（-0.3h）不大

############# C4 ################

data = woo.dataWoo('WAGE2')

# IQ来解释月薪wage
y = data['wage']
x = data['IQ']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model_summary = model.summary()
print(model_summary)

# i 平均工资和平均iq ，IQ的标准差

IQ_average = data['IQ'].mean()
IQ_std = data['IQ'].std()
wage_average = data['wage'].mean()

#  8.3031 // 120 // 9.6%_不能

#  13.2%
y = data['lwage']
x = data['IQ']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model_summary = model.summary()
print(model_summary)
result = (0.0088 * 15) * 100

############# C5 ################

data = woo.dataWoo('RDCHEM')

# rd 关于sales的弹性估值： 1.0757%

y = data['lrd']
x = data['lsales']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model_summary = model.summary()
print(model_summary)

############ C5 ################

data = woo.dataWoo('ATTEND')

# i 最大值最小值
atndrte_max = data['atndrte'].max()
atndrte_min = data['atndrte'].min()

# ii 建模  -0.7637  明显的
y = data['atndrte']
x = data['ACT']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model_summary = model.summary()
print(model_summary)



