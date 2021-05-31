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


