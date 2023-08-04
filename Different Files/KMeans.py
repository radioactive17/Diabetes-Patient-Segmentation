#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA as sPCA
plt.style.use('ggplot')


# In[2]:


df = pd.read_csv('diabetic_data.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


df = df.drop(['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code'], axis = 1)
df


# In[5]:


for itr, col in enumerate(df.columns):
    print(f"{col}\n{df[col].unique()}\n")


# In[6]:


df = df.drop(['examide', 'citoglipton'], axis = 1)
df


# In[7]:


for itr, col in enumerate(df.columns):
    try:
        x = df[col].str.contains('\?').sum()
        print(f"{col} - {x}")
    except:
        pass


# In[8]:


df['medical_specialty'] = df['medical_specialty'].replace('?', 'Unknown')
df['weight'] = df['weight'].replace('?', np.mean(pd.to_numeric(df['weight'], errors='coerce')))


# In[9]:


for itr, col in enumerate(df.columns):
    try:
        df = df[~df[col].str.contains("\?")]
        #x = df[col].str.contains('\?').sum()
        #print(f"{col} - {x}")
    except:
        pass


# In[10]:


for itr, col in enumerate(df.columns):
    print(f"{col}\n {df[col].value_counts()}\n")


# In[11]:


df = df.drop(['metformin-rosiglitazone'], axis = 1)


# In[12]:


objs = df.loc[:, df.dtypes == 'object'].columns.tolist()
le = LabelEncoder()
for col in objs:
    df[col] = le.fit_transform(df[col])
df


# In[13]:


X = df.iloc[:,:-1]
Y = df.iloc[:,-1]


# In[14]:


X_norm = (X-X.mean())/(X.max()-X.min())
X_norm


# In[15]:


X_norm = X_norm.to_numpy()
X_norm


# In[16]:


X_norm.shape


# In[17]:


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
        print(scalar_product)
    return centers


# In[18]:


Ck(X_norm, 'euclidean', 3, 0)


# In[ ]:




