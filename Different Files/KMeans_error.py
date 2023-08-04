#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nump_indexed


# In[2]:


pip install polars


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA as sPCA
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import numpy_indexed as npi
import time
import polars as pl


# ### Diabetes Dataset

# In[4]:


df = pd.read_csv('diabetic_data.csv')
df


# In[5]:


df.isnull().sum()


# In[6]:


df = df.drop(['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code'], axis = 1)
df


# In[7]:


for itr, col in enumerate(df.columns):
    if len(df[col].unique().tolist()) == 1:
        df = df.drop([col], axis = 1)
df


# In[8]:


for itr, col in enumerate(df.columns):
    try:
        x = df[col].str.contains('\?').sum()
        print(f"{col} - {x}")
    except:
        pass


# In[9]:


df = df.drop(['weight'], axis = 1)
df


# In[10]:


df['medical_specialty'] = df['medical_specialty'].replace('?', 'Unknown')
df = df.replace('?', np.nan)
df.isna().sum()


# In[11]:


df = df.dropna()
df


# In[12]:


for itr, col in enumerate(df.columns):
    if len(df[col].unique().tolist()) == 1:
        df = df.drop([col], axis = 1)
df


# In[13]:


X = df.iloc[:,:-1]
Y = df.iloc[:,-1]


# In[14]:


Y.unique()


# In[15]:


y = Y.replace({'NO': 0, '<30': 1, '>30': 1})
y_np = y.to_numpy()
y_df = pd.DataFrame(y_np)
y_df


# In[16]:


objs = X.loc[:, X.dtypes == 'object'].columns.tolist()
le = LabelEncoder()
for col in objs:
    X[col] = le.fit_transform(X[col])
X


# In[17]:


scaler = MinMaxScaler()
X_norm = scaler.fit(X)
X_norm = scaler.transform(X)
X_norm
# X_norm = (X-X.min())/(X.max()-X.min())
# X_train = X_norm.to_numpy()
# X_train


# In[18]:


X_norm.shape


# In[19]:


def assign_center(X, k):
    centers = list()
    for i in range(k):
        centers.append(X[np.random.randint(X.shape[0], size = 1)])
    return centers

def assign_clusters(data, centers, k):
    clusters = dict()
    for i in range(k):
        clusters[i] = list()
        
    for itr, x in enumerate(data):
        distance_list = list()
        for j in range(k):
            dist = np.linalg.norm(x - centers[j])
            distance_list.append(dist)
        clusters[distance_list.index(min(distance_list))].append(x)
    return clusters

def calculate_centers(clusters, centers, k):
    for i in range(k):
        centers[i] = np.average(clusters[i], axis = 0)
    return centers

def calculate_scalar(prev_centers, new_centers, k):
    total = 0
    for i in range(k):
        total += np.linalg.norm(prev_centers[i] - new_centers[i])
    return total/k

def Ck(data, distance, k, threshold):
    #Initializing centroids
    centers = list()
    for i in range(k):
        centers.append(data[np.random.randint(data.shape[0], size = 1)])
    
    #KMeans 
    scalar_product = 10e7
    while scalar_product > threshold:
        previous_centers = centers.copy()
        clusters = assign_clusters(data, centers, k)
        centers = calculate_centers(clusters, centers, k)
        scalar_product = calculate_scalar(previous_centers, centers, k)
        #print(scalar_product)
    return centers, clusters


# In[20]:


st = time.time()
centers, clusters = Ck(X_norm, 'euclidean', 2, 0)
ed = time.time() - st
ed


# In[21]:


error_rate_dict = dict()
time_dict = dict()
for k in range(2,6):
    error_rate_dict[k] = list()
    time_dict[k] = list()
    
for _ in range(20):
    for k in range(2,6):
        st = time.time()
        centers, clusters = Ck(X_norm, 'euclidean', k, 0)
        error_rate = 0
        for i in range(k):
            result = npi.indices(X_norm, clusters[i])
            result_df = pd.DataFrame(result)
            yi = (y_df[0][result_df[0]] == 1).sum()
            ni = (y_df[0][result_df[0]] == 0).sum()
            if yi > ni:
                error_rate += ni/(yi+ni)
            else:
                error_rate += yi/(ni+yi)
        error_rate_dict[k].append(round(error_rate/k, 2))
        time_dict[k].append(time.time() - st)
    print(f"{error_rate_dict}\n")
    print(f"{time_dict}\n")


# In[22]:


fig, axis = plt.subplots()
x_labels = list(error_rate_dict.keys())
axis.boxplot(error_rate_dict.values())
axis.set_xticklabels(x_labels)
axis.set_title('Distribution of errors in Standard K-Means - Diabetes Data')


# In[23]:


fig, axis = plt.subplots()
x_labels = list(time_dict.keys())
axis.boxplot(time_dict.values())
axis.set_xticklabels(x_labels)
axis.set_title('Distribution of time in Standard K-Means - Diabetes Data')


# In[ ]:




