import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans

df = pd.read_csv('diabetic_data.csv')

#print(f"Columns:\n{df.columns}")
#print(f"Dtypes:\n{df.dtypes}")
#print(f"Shape of dataset:\n{df.shape}")

#PREPROCESSING OF THE DATASET
print(df.shape)

#Dropping columns that have more than 75% missing/null values
for itr, col in enumerate(df.columns):
    if df[col].isnull().sum() > df.shape[0]*0.75:
        df = df.drop([col], axis = 1)

#Dropping irrelevant columns
df = df.drop(['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code'], axis = 1)

#Dropping columns that have only single value in them
for itr, col in enumerate(df.columns):
    if len(df[col].unique().tolist()) == 1:
        #print(f"{col}\n{df[col].unique()}\n")
        df = df.drop([col], axis = 1)

#Finding columns which contains '?' in them
for itr, col in enumerate(df.columns):
    try:
        x = df[col].str.contains('\?').sum()
        print(f"{col} - {x}")
    except:
        pass

#Replacing or Dropping columns that contain '?' in them with suitable values
df = df.drop(['weight'], axis = 1)
df['medical_specialty'] = df['medical_specialty'].replace('?', 'Unknown')
df = df.replace('?', np.nan)
df = df.dropna()

#Checking again for columns that have only single value in them
for itr, col in enumerate(df.columns):
    if len(df[col].unique().tolist()) == 1:
        df = df.drop([col], axis = 1)


X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

y = Y.replace({'NO': 0, '<30': 1, '>30': 1})
y_np = y.to_numpy()
y_df = pd.DataFrame(y_np)

objs = X.loc[:, X.dtypes == 'object'].columns.tolist()
le = LabelEncoder()
for col in objs:
    X[col] = le.fit_transform(X[col])

scaler = MinMaxScaler()
X_norm = scaler.fit(X)
X_norm = scaler.transform(X)
