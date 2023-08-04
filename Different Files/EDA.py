#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


# In[3]:


df = pd.read_csv('diabetic_data.csv')
df


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[10]:


#Total entries in the dataset
df.shape


# In[13]:


#Unknown or missing data in the dataset
df.isnull().sum().sum()


# In[39]:


#Histogram for attributes age, num_lab_procedures, and num_medications
cols = ['age', 'num_lab_procedures', 'num_medications']
fig, axis = plt.subplots(1, 3, figsize = (25, 8))
axis = axis.ravel()
for itr, col in enumerate(cols):
    plt.subplot(1, 3, itr+1)
    sns.histplot(df[col], kde=True, stat="density", linewidth=1)


# In[64]:


plt.figure(figsize = (15, 7))
print(df['readmitted'].unique())
markers = {'NO': 'X', '>30': 'o', '<30': 's'}
sns.scatterplot(data = df, x='num_medications', y='time_in_hospital', hue = 'readmitted', style = 'readmitted', markers = markers)


# In[66]:


plt.figure(figsize = (15, 7))
sns.scatterplot(data = df, x='num_lab_procedures', y='num_medications', hue = 'readmitted', style = 'readmitted', markers = markers)


# In[ ]:




