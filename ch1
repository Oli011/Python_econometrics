# -*- coding: utf-8 -*-
"""
chapter 1 
"""
import pandas as pd
import wooldridge as woo
import numpy as np

data = woo.dataWoo('WAGE1')

############# C1 ################
# i 求平均值、最小值和最大值
eudc_average = data['educ'].mean()
eudc_max = data['educ'].max()
eudc_min = data['educ'].min()

# ii 求wage均值
wage_average = data['wage'].mean()

# iii 两年CPI
# CPI in 1976 is 56.9 (average), CPI in 2010 is 218.056

# iv 平均工资现值
wage_average_now = (data.wage * ((2.18056 / 0.569))).mean()

# v 女人和男人数量
female = data['female'].sum()
# 返回形状，即几行几列的数组，如[2,3],shape[0]=rows,shape[1]=columns
male = data.shape[0] - female

############# C2 ################
# 导入对应数据集
data2 = woo.dataWoo('BWGHT2')

# i 妇女数量、怀孕期间抽烟数量
female2 = data2.shape[0]
some = data2[(data2['cigs'] > 0)]
cig_female2 = some.shape[0]

# # 生男孩妇女数量、生男孩报告怀孕期间吸烟
# male2 = data2['male'].sum()
# female2 = data2.shape[0] - male2
# 多重条件筛选后再形成dataframe
# some = data2[(data2['male'] == 0) & (data2['cigs'] > 0)]
# cig_female2 = some.shape[0]

# ii 平均每天抽烟数量 平均数作为度量：不好，吸烟的人很少
averagecig_total = data2['cigs'].mean()

# iii 怀孕期间妇女的平均抽烟量
averagecig_cig = some['cigs'].mean()

# iv feduc 的平均值
have_fa = data2[(data2['feduc'] > 0)]
feduc_average = have_fa['feduc'].mean()

# v npvis的众数值
mode = data2['npvis'].mode()
############# C3 ################
# 导入对应数据集
data = woo.dataWoo('MEAP01')
# i math4 的最大值和最小值 不合理
math_max = data['math4'].max()
math_min = data['math4'].min()

# ii 数学测试100%通过
math_100 = data[data['math4'] == 100]
math_100_num = math_100.shape[0]
math_100_percent = (math_100_num / data.shape[0]) * 100

# iii 数学通过率50%
math_50 = data[data['math4'] == 50]
math_50_num = math_50.shape[0]

# iv 数学和阅读的平均通过率 阅读更难
math_average = data['math4'].mean()
reading_average = data['read4'].mean()

# v 相关系数
corr = data['math4'].corr(data['read4'])

# vi exppp的平均值和标准差 变动不大
exppp_average = data['exppp'].mean()
exppp_std = data['exppp'].std()

# vii  A对B支出超过百分比
result = (6000 / 5500 - 1) * 100
final_result = 100 * (np.log(6000) - np.log(5500))

############# C4 ################
导入对应数据集
data = woo.dataWoo('JTRAIN2')

# i train变量得到工作的男性比例
train_data = data[data['train'] == 1 ]
percent = (train_data.shape[0] / data.shape[0]) * 100
# 或者，直接求数量
# data['train'].sum() / data.shape[0]


# ii re78 平均值 差别不大
untrain_data = data[data['train'] == 0 ]
train_average = train_data['re78'].mean()
untrain_average = untrain_data['re78'].mean()

# iii 78失业比例 未培训的失业比例更高
un78_data = train_data[train_data['unem78'] == 1]
un_percent = (un78_data.shape[0] / train_data.shape[0]) * 100

un78_untrain_data = untrain_data[untrain_data['unem78'] == 1]
untrain_un_percent = (un78_untrain_data.shape[0] / untrain_data.shape[0]) * 100

iv 工作培训项目是否有效, 
在注意到失业率差异之前，
工作培训似乎是有效的。在令人信服之前，结果需要针对失业进行标准化。
