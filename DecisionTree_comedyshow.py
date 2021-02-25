# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:09:48 2021

@author: aai00920
"""

import os
import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

os.chdir(r'C:\Users\AAI00920\OneDrive - ARCADIS\Desktop')
path = r'C:\Users\AAI00920\OneDrive - ARCADIS\Desktop'
df = pd.read_excel('comedyshow_data.xlsx')

#To make a decision tree, all data has to be numerical. We have to convert the non numerical columns 'Nationality' and 'Go' into numerical values. Pandas has a map() method that takes a dictionary with information on how to convert the values.

d = {'UK': 0, 'USA': 1, 'N' : 2}
df['Nationality'] = df['Nationality'].map(d)

d = {'YES': 0, 'NO' : 1}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

print(X)
print(y)

#Create a Decision Tree, save it as an image, and show the image:

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

target_names = ["Go", "No Go"]
y_pred = dtree.predict(X)
from sklearn.metrics import classification_report
print(classification_report(y, y_pred, target_names=target_names))

#to predict on new data on a new excel
df1 = pd.read_excel('comedyshow_data1.xlsx')

d = {'UK': 0, 'USA': 1, 'N' : 2}
df1['Nationality'] = df1['Nationality'].map(d)

X_new = df1[['Age', 'Experience', 'Rank', 'Nationality']]
target_names = ["Go", "No Go"]
y_pred = dtree.predict(X_new)






