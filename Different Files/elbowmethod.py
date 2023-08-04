#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yellowbrick


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA as sPCA
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy_indexed as npi


# In[3]:


df = pd.read_csv('diabetic_data.csv')
df


# In[4]:


df.isnull().sum()


# In[5]:


df = df.drop(['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code'], axis = 1)
df


# In[6]:


for itr, col in enumerate(df.columns):
    if len(df[col].unique().tolist()) == 1:
        df = df.drop([col], axis = 1)
df


# In[7]:


for itr, col in enumerate(df.columns):
    try:
        x = df[col].str.contains('\?').sum()
        print(f"{col} - {x}")
    except:
        pass


# In[8]:


df = df.drop(['weight'], axis = 1)
df


# In[9]:


df['medical_specialty'] = df['medical_specialty'].replace('?', 'Unknown')
df = df.replace('?', np.nan)
df.isna().sum()


# In[10]:


df = df.dropna()
df


# In[11]:


for itr, col in enumerate(df.columns):
    if len(df[col].unique().tolist()) == 1:
        df = df.drop([col], axis = 1)
df


# In[12]:


X = df.iloc[:,:-1]
Y = df.iloc[:,-1]


# In[13]:


Y.unique()


# In[14]:


y = Y.replace({'NO': 0, '<30': 1, '>30': 1})
y_np = y.to_numpy()
y_df = pd.DataFrame(y_np)
y_df


# In[15]:


objs = X.loc[:, X.dtypes == 'object'].columns.tolist()
le = LabelEncoder()
for col in objs:
    X[col] = le.fit_transform(X[col])
X


# In[16]:


scaler = MinMaxScaler()
X_norm = scaler.fit(X)
X_norm = scaler.transform(X)
X_norm
# X_norm = (X-X.min())/(X.max()-X.min())
# X_train = X_norm.to_numpy()
# X_train


# In[17]:


X_norm.shape


# In[21]:


km = KMeans()
visualizer = KElbowVisualizer(km, k=(2,15))
visualizer.fit(X_norm)
visualizer.show()


# In[ ]:





# In[ ]:




