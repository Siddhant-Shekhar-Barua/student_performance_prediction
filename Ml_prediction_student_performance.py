# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:02:19 2025

@author: HP
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import accuracy_score , r2_score ,  mean_squared_error


df = pd.read_csv(r'C:\Users\HP\Documents\python\Student_performance.csv')
print(df)


print('EDA:')
print('dimensions:',df.shape)
print('descriptive_stats:') 
print(df.describe)
print(df.isnull())
print('top 10 values:')
print(df.head(10))
print('bottom 10 values:')
print(df.tail(10))

print(df.columns)

df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

X = df.drop('Performance Index', axis=1)  # Features (input)
y = df['Performance Index']

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state= 42)

model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R-squared Score:", r2_score(y_test, y_pred_lr))







