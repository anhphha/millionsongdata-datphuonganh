import numpy as np
import pandas as pd
import re
import os
import sys
import glob
import copy
import sqlite3
import multiprocessing

data = pd.read_csv('SongCSV.csv')


print(len(data))

data['Year'] = data['Year'].replace(0, np.nan)
data['SongHotness'] = data['SongHotness'].replace(0, np.nan) 
data.dropna(subset=['Year','SongHotness'], inplace=True)
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


