## TASK 1: DATA EXPLORATION 

import numpy as np
import pandas as pd
import re
import time
import os
import sys
import glob
import copy
import sqlite3
import multiprocessing

data = pd.read_csv('SongCSV.csv')

start_time = time.time()

print(len(data))

data['SongHotness'] = data['SongHotness'].replace(0, np.nan) 
data.dropna(subset=['SongHotness'], inplace=True)
data['SongID'] = data['SongID'].replace('b','',regex=True)
data['SongID'] = data['SongID'].replace('\'','',regex=True)
data['ArtistID'] = data['ArtistID'].replace('b','',regex=True)
data['ArtistID'] = data['ArtistID'].replace('\'','',regex=True)
data['ArtistName'] = data['ArtistName'].replace('b','',regex=True)
data['ArtistName'] = data['ArtistName'].replace('\'','',regex=True)
data['Title'] = data['Title'].replace('b','',regex=True)
data['Title'] = data['Title'].replace('\'','',regex=True)
data['AudioMd5'] = data['AudioMd5'].replace('b','',regex=True)
data['AudioMd5'] = data['AudioMd5'].replace('\'','',regex=True)
data.reset_index(drop=True, inplace=True)

#Data plotting
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['SongID']=le.fit_transform(data.SongID)
data['ArtistID']=le.fit_transform(data.ArtistID)
data['AudioMd5']=le.fit_transform(data.AudioMd5)
data['ArtistName']=le.fit_transform(data.ArtistName)
data['Title']=le.fit_transform(data.Title )

print(len(data))  
  
import matplotlib.pyplot as plt
from matplotlib import rc
font = {'family' : 'monospace', 'weight' : 'bold', 'size'   : 10}
rc('font', **font) 
plt.rcParams['figure.figsize'] = [20, 15]
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 5.0

def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

draw_histograms(data, data.columns, 5, 4)
plt.savefig('data_plot.png', bbox_inches='tight')
plt.close(1)

#Feature exploration
data.drop(['SongID','ArtistID', 'ArtistName', 'Title','AudioMd5', 'Year','SongNumber'], axis=1, inplace=True)
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


## TASK 2: MODE PREDICTION 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

data = pd.read_csv('SongCSV.csv')
start_time = time.time()

data['SongHotness'] = data['SongHotness'].replace(0, np.nan)  
data.dropna(subset=['SongHotness'], inplace=True)
data.drop(['SongID','ArtistID', 'ArtistName', 'Title','AudioMd5', 'Year' ], axis=1, inplace=True)
data.reset_index(drop=True, inplace=True)

X = data.loc[:,data.columns!='mode']
print(len(X))
y = data['mode']
print(len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#SVC 


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
print("Generating SVC...")

svm = SVC()

param_grid1 = {'C': [0.001, 0.01, 0.1],
              'gamma': [0.001, 0.01, 0.1]          
             }

CV1 = GridSearchCV(svm, 
                  param_grid1,
                  scoring='accuracy',
                  return_train_score=True,
                  n_jobs=-1,
		  cv=5)

CV1.fit(X_train, y_train)

print('best params: ', CV1.best_params_)    
print('best score: %.3f' % CV1.best_score_)
print('accuracy score: %.3f' % CV1.score(X_test, y_test))

pred1 = CV1.predict(X_test)
ac1 = accuracy_score(y_test,pred1)
print('Accuracy is: ',ac1)


# KNN 

from sklearn.neighbors import KNeighborsClassifier

print("Generating kNN...")

knn = KNeighborsClassifier() 

param_grid2 = {'n_neighbors': np.arange(4,100)}


CV2 = GridSearchCV(knn, 
                  param_grid2,
                  return_train_score=True, 
                  n_jobs=-1,
		  cv=5)

CV2.fit(X_train, y_train)

print('best params: ', CV2.best_params_)    
print('best score: %.3f' % CV2.best_score_)
print('accuracy score: %.3f' % CV2.score(X_test, y_test))

pred2 = CV2.predict(X_test)
ac2 = accuracy_score(y_test,pred2)
print('Accuracy is: ',ac2)

#Random Forest Classification

from sklearn.ensemble import RandomForestClassifier


print("Generating Random Forest Classifier...")

rfc = RandomForestClassifier()

param_grid3 = {'n_estimators': [50, 100, 200],
               'max_depth': [10, 20, 30],
              'criterion': ['gini'],
              }

CV3 = GridSearchCV(rfc, 
                  param_grid3,
                  scoring='accuracy',
                  return_train_score=True,
                  n_jobs=-1,
		  cv=5)

CV3.fit(X_train, y_train)
print('best params: ', CV3.best_params_)    
print('best score: %.3f' % CV3.best_score_)
print('accuracy score: %.3f' % CV3.score(X_test, y_test))

pred3 = CV3.predict(X_test)
ac3 = accuracy_score(y_test,pred3)
print('Accuracy is: ',ac3)

#SGD 

from sklearn.linear_model import SGDClassifier

print("Generating SGD...")


sgdc = SGDClassifier()

param_grid4 = {'loss': [ "log"],
              'penalty': ["l1", "l2", "elasticnet"],
              'max_iter': [10000,100000,1000000], 
              'alpha': [0.0001, 0.01, 0.1, 1]
              }

CV4 = GridSearchCV(sgdc, 
                  param_grid4,
                  scoring='accuracy',
                  return_train_score=True,
                  n_jobs=-1,
		  cv=5)

CV4.fit(X_train, y_train)

print('best params: ', CV4.best_params_)    
print('best score: %.3f' % CV4.best_score_)
print('accuracy score: %.3f' % CV4.score(X_test, y_test))

pred4 = CV4.predict(X_test)
ac4 = accuracy_score(y_test,pred4)
print('Accuracy is: ',ac4)

d = {'Model':['SVC','KNN','Random Forest','SGD'],'Accuracy':[ac1, ac2, ac3, ac4]}

agg_scores = pd.DataFrame(data = d)
agg_scores.plot(x='Model',y='Accuracy',kind='bar')
plt.savefig('model_comparison.png', bbox_inches='tight')
plt.close(1)

print("--- %s seconds ---" % (time.time() - start_time)) 

## TASK 3: ARTIST RECOMMENDATION 

import numpy as np
import pandas as pd
import re
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np 

data = pd.read_csv('SongCSV.csv') 

data['SongID'] = data['SongID'].replace('b','',regex=True)
data['SongID'] = data['SongID'].replace('\'','',regex=True)
data['ArtistID'] = data['ArtistID'].replace('b','',regex=True)
data['ArtistID'] = data['ArtistID'].replace('\'','',regex=True)
data['ArtistName'] = data['ArtistName'].replace('b','',regex=True)
data['ArtistName'] = data['ArtistName'].replace('\'','',regex=True)
data['Title'] = data['Title'].replace('b','',regex=True)
data['Title'] = data['Title'].replace('\'','',regex=True)

data['Song'] = data["ArtistName"] + ' - ' + data["Title"]
song_grouped = data.groupby(['Song']).agg({'SongNumber': 'count'}).reset_index()
grouped_sum = song_grouped['SongNumber'].sum()
song_grouped['percentage']  = song_grouped['SongNumber'].div(grouped_sum)*100
song_grouped.sort_values(['SongNumber', 'Song'], ascending = [0,1])

artists = data['ArtistID'].unique()
print(len(artists))

# from Recommenders import Recommenders
train_data, test_data = train_test_split(data, test_size = 0.20, random_state=0)


#Class for Popularity based Recommender System model
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.ArtistID = None
        self.SongID = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, ArtistID, SongID):
        self.train_data = train_data
        self.ArtistID = ArtistID
        self.SongID = SongID

        #Get a count of ArtistID for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.SongID]).agg({self.ArtistID: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'ArtistID': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.SongID], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to make recommendations
    def recommend(self, ArtistID):    
        artist_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        artist_recommendations['ArtistID'] = ArtistID
    
        #Bring user_id column to the front
        cols = artist_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        artist_recommendations = artist_recommendations[cols]
        
        return artist_recommendations

pm = popularity_recommender_py()
pm.create(train_data, 'ArtistID', 'Song')

#user the popularity model to make some prediction
ArtistID = artists[8]
print(pm.recommend(ArtistID))

## TASK 4: YEAR PREDICTION 

import numpy as np 
import pandas as pd

data = pd.read_csv('SongCSV.csv')
data['Year'] = data['Year'].replace(0, np.nan)
data['SongHotness'] = data['SongHotness'].replace(0, np.nan)  
data.dropna(subset=['Year', 'SongHotness'], inplace=True)
data = data[data.Year>1990]

data.drop(['SongID','ArtistID', 'ArtistName', 'Title','AudioMd5'], axis=1, inplace=True)

nsongs = {}
for y in range(1991,2009):
    nsongs[y] = len(data[data.Year==y])
    print("Year=%d, nsongs=%d" % (y, nsongs[y]))

X = data.loc[:,data.columns!='Year']
y = data.Year

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test,  y_train, y_test = train_test_split(X_scaled,y,train_size=0.8, random_state=0)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', n_jobs=None, random_state=None)
forest.fit(X_train, y_train)
print("Score of training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Score of test set: {:.3f}".format(forest.score(X_test, y_test)))

y_pred = forest.predict(X_test)
y_pred

from sklearn.metrics import classification_report,confusion_matrix
print("Classification Report")
print(classification_report(y_pred, y_test))
labels = np.unique(y_test)
print("Confusion Matrix")
a = confusion_matrix(y_pred, y_test, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)

names = list(data.columns)
print(names)
X_names = names
def plot_feature_importances(model):
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center', color='orange', label=50)
    plt.yticks(np.arange(n_features), X_names)
    plt.title('Feature importance for dataset')
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

from matplotlib import pyplot as plt
plot_feature_importances(forest)

from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=2000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
print(sgdc)
 
sgdc.fit(X_train, y_train)
print("Score of training set: {:.3f}".format(sgdc.score(X_train, y_train)))
print("Score of test set: {:.3f}".format(sgdc.score(X_test, y_test)))

y_pred_sgd = sgdc.predict(X_test)

cr = classification_report(y_test, y_pred_sgd)
print(cr)

cm = confusion_matrix(y_test, y_pred_sgd)
print(cm)

## TASK 5: SONG POPULARITY 

import numpy as np
import pandas as pd

data = pd.read_csv('SongCSV.csv')

data['Year'] = data['Year'].replace(0, np.nan)
data['SongHotness'] = data['SongHotness'].replace(0, np.nan)    

data.dropna(subset=['Year', 'SongHotness'], inplace=True)
data = data[data.Year>1990]

data['SongID'] = data['SongID'].replace('b','',regex=True)
data['SongID'] = data['SongID'].replace('\'','',regex=True)
data['ArtistID'] = data['ArtistID'].replace('b','',regex=True)
data['ArtistID'] = data['ArtistID'].replace('\'','',regex=True)
data['ArtistName'] = data['ArtistName'].replace('b','',regex=True)
data['ArtistName'] = data['ArtistName'].replace('\'','',regex=True)
data['Title'] = data['Title'].replace('b','',regex=True)
data['Title'] = data['Title'].replace('\'','',regex=True)
data['AudioMd5'] = data['AudioMd5'].replace('b','',regex=True)
data['AudioMd5'] = data['AudioMd5'].replace('\'','',regex=True)

data.groupby(data.Year).mean().reset_index()

song_hotness_mean = data.SongHotness.mean()
threshold = song_hotness_mean
data.loc[data.SongHotness>=threshold,'isPopular'] = 1
data.loc[data.SongHotness<threshold,'isPopular'] = 0

#Most Popular Songs
data.sort_values(by=['SongHotness','Year'],ascending=False)[['Title','ArtistName','SongHotness','Year']].head(10)

from sklearn.ensemble import RandomForestRegressor
X = data.drop(['SongHotness','Year','isPopular'],axis=1)
X = X.select_dtypes(exclude=[object])
Y = data['SongHotness'].values

model = RandomForestRegressor(n_estimators=100,max_depth=40)
model.fit(X,Y)

features = X.columns
feature_importances = model.feature_importances_

features_data = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
features_data.sort_values('Importance Score', inplace=True, ascending=False)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = data.drop('SongHotness',axis=1)
X = X.select_dtypes(exclude=[object])
X_top_5_features = X[list(features_data.head().Features)] 
Y = data['isPopular'].values

trainX, testX, trainY, testY = train_test_split(X_top_5_features, Y, train_size=0.70)

knn = KNeighborsClassifier()
knn.fit(trainX,trainY)

pred1 = knn.predict(testX)

print('Score of knn model: %.3f' % knn.score(testX, testY))
ac1 = accuracy_score(testY,pred1)
print('Accuracy of knn model:',ac1)

#SVC 
from sklearn.svm import SVC
svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',random_state=None)
svm.fit(trainX,trainY)
pred2= svm.predict(testX)
print('Score of SVC model: %.3f' % svm.score(testX, testY))
ac2 = accuracy_score(testY,pred2)
print('Accuracy of SVC model:',ac2)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,criterion='gini', max_depth=None, min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',random_state=None)

rf.fit(trainX,trainY)
pred3= rf.predict(testX)
print('Score of random forest model: %.3f' % rf.score(testX, testY))
ac3 = accuracy_score(testY,pred3)
print('Accuracy of Random forest model: ',ac3)

#SGD

from sklearn.linear_model import SGDClassifier

sgdc = SGDClassifier()
sgdc.fit(trainX,trainY)
pred4= sgdc.predict(testX)
print('Score of SGDClassifier model: %.3f' % sgdc.score(testX, testY))
ac4 = accuracy_score(testY,pred4)
print('Accuracy of SGDClassifier model: ',ac4)

## TASK 6: SONG RECOMMENDATION

import numpy as np
import pandas as pd

data = pd.read_csv('SongCSV.csv')


data['SongID'] = data['SongID'].replace('b','',regex=True)
data['SongID'] = data['SongID'].replace('\'','',regex=True)
data['ArtistID'] = data['ArtistID'].replace('b','',regex=True)
data['ArtistID'] = data['ArtistID'].replace('\'','',regex=True)
data['ArtistName'] = data['ArtistName'].replace('b','',regex=True)
data['ArtistName'] = data['ArtistName'].replace('\'','',regex=True)
data['AudioMd5'] = data['AudioMd5'].replace('b','',regex=True)
data['AudioMd5'] = data['AudioMd5'].replace('\'','',regex=True)
data['Title'] = data['Title'].replace('b','',regex=True)
data['Title'] = data['Title'].replace('\'','',regex=True)
data['Title'] = data['Title'].replace('(, )','',regex=True)
print(len(data))

data['Song'] = data["SongID"] + ' - ' + data["Title"]
song_grouped = data.groupby(['Song']).agg({'SongID': 'count'}).reset_index()
grouped_sum = song_grouped['SongID'].sum()
song_grouped.sort_values(['SongID', 'Song'], ascending = [0,1])
song_grouped.rename(columns = {'SongID': 'Songcount'},inplace=True)

artists = data['ArtistID'].unique()
print(len(artists))

import sklearn
from sklearn.model_selection import train_test_split 

train_data, test_data = train_test_split(data, test_size = 0.20, random_state=0)

import numpy as np

class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.ArtistID = None
        self.SongID = None
        self.popularity_recommendations = train_data

    def recommend(self, ArtistID):    
        artist_recommendations = self.popularity_recommendations
        artist_recommendations = artist_recommendations[artist_recommendations['ArtistID'] == ArtistID]
        artist_recommendations = artist_recommendations.loc[:,['Title','ArtistName']]
        return artist_recommendations

pm = popularity_recommender_py()
ArtistID = artists[6]
print(pm.recommend(ArtistID))