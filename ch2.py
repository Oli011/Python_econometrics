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

# iii 解释方程中的截距和mrate的系数
# 遇到了知识性的障碍，打算先看看书再进行代码练习
