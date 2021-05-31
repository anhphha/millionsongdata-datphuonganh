import numpy as np
import pandas as pd
import re
import time

data = pd.read_csv('SongCSV.csv')

start_time = time.time()

print(len(data))

data['SongHotness'] = data['SongHotness'].replace(0, np.nan) 
data.dropna(subset=['SongHotness'], inplace=True)
data.drop(['SongID','ArtistID', 'ArtistName', 'Title','AudioMd5', 'Year','SongNumber'], axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)
data.to_csv('SongCSV_clean.csv')
print(data.head(10))

print(len(data)) 


X = data.loc[:,data.columns!='mode']
print(X[:10])

y = data['mode']
print(y[:10]) 

import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import math

ax = sns.countplot(y, label="Count", palette='seismic')
Y, N = y.value_counts()
print('Number of mode 1: ', Y)
print('Number of mode 0 : ', N)

#correlation map  
f,ax = plt.subplots(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
plt.savefig('correlation_map.png', bbox_inches='tight')
plt.close(1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, random_state=None) 
    
clr_rf = clf_rf.fit(X_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(X_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(X_test))
sns.set(rc={'figure.figsize':(10,5)}) 
sns.heatmap(cm,annot=True,fmt="d")
plt.savefig('train_test_accuracy.png', bbox_inches='tight')
plt.close(2)

clf_rf_1 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, random_state=None)      
clr_rf_1 = clf_rf_1.fit(X_train,y_train)
importances = clr_rf_1.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(15, 10))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices],rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.close(3)

print("--- %s seconds ---" % (time.time() - start_time))


