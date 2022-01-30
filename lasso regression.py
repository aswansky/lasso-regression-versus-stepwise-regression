# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:13:27 2022

@author: abdal
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
import os


data=pd.read_csv(r'...\SampleSuperstore.csv')
data = data.drop(['Row ID','Order ID','Customer ID','Customer Name', 'Product ID'], axis=1)
def func(x):
    day, month,year=x.split('/')
    return day, month,year

#proprocessing steps
data_dates=pd.DataFrame([[a, b, c] for a,b,c in data['Order Date'].apply(lambda x:func(x)).values],
                        columns=['order_day','order_month','order_year'])
data=data.drop(['Order Date'],axis=1)
data=pd.concat([data,data_dates],axis=1)

data_dates1=pd.DataFrame([[a, b, c] for a,b,c in data['Ship Date'].apply(lambda x:func(x)).values],
                        columns=['ship_day','ship_month','ship_year'])
data=data.drop(['Ship Date'],axis=1)
data=pd.concat([data,data_dates1],axis=1)
encoded_data = pd.get_dummies(data,columns=['Ship Mode','Segment','Country','City','State','Postal Code','Region',
                                        'Category','Sub-Category','Product Name','Quantity','order_day','order_month','order_year','ship_day','ship_month','ship_year'])
data_x0=encoded_data[[x for x in encoded_data.columns if x not in ('Profit')]]
data_y=data['Profit']
data_y=data_y.values.reshape(-1,1)
datacsv=pd.concat([data_x0,pd.DataFrame(data_y)],axis=1)
sch=StandardScaler()
data_x=StandardScaler().fit_transform(data_x0)
lassoreg=Lasso().fit(data_x,data_y)
y_predicted=lassoreg.predict(data_x)
y_predicted=y_predicted.reshape(-1,1)
r2=r2_score(data_y, y_predicted)
coeff=lassoreg.coef_

all_coeff=[]
for i,coefficient in enumerate(coeff):
    if coefficient!=0:
        coeff_d={}
        coeff_d['feature']=data_x0.columns[i]
        coeff_d['coeff']=coefficient
        all_coeff.append(coeff_d)
    
all_coeff1=pd.DataFrame(all_coeff)
all_coeff1_sorted=all_coeff1.sort_values('coeff')
lowest_coeff=all_coeff1_sorted.head(10)
highest_coeff=all_coeff1_sorted.tail(10)
pd.set_option('display.max_colwidth', -1)

ax=sb.barplot(x='feature',y='coeff',data=pd.concat([lowest_coeff,highest_coeff],axis=0))
for item in ax.get_xticklabels():
    item.set_rotation(90)