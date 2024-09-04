#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('CAR DETAILS.csv')
df.head()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.dtypes
df.shape
cols=df.columns
cols
cat_cols=df.dtypes[df.dtypes=='object'].index
num_cols=df.dtypes[df.dtypes!='object'].index
print(cat_cols)
print(num_cols)
a =  df[num_cols].describe(percentiles=[0.01,0.02,0.03,0.25,0.5,0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99]).T
a
df = df[(df['selling_price'] > 0) & (df['km_driven'] > 0)]
print(df.describe())
print('Year : ',df.year.unique(),'\n')
print('fuel :', df.fuel.unique(), '\n')
print('Transmission : ', df.transmission.unique(), '\n')
print('seller Type: :', df.seller_type.unique(),'\n')
print('Owner :', df.owner.unique(), '\n')
df['brand']=df['name'].str.split(expand=True)[0]
df['model']=df['name'].str.split(expand=True)[1]
df.head()
df.to_csv('car_data_clean.csv')
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['fuel_type'] = lb.fit_transform(df['fuel'])
df['Type_of_Seller'] = lb.fit_transform(df['seller_type'])
df['Transmit'] = lb.fit_transform(df['transmission'])
df['Owner_Type'] = lb.fit_transform(df['owner'])
df['brand_name'] = lb.fit_transform(df['brand'])
df['model_name'] = lb.fit_transform(df['model'])
cols = ['year','fuel', 'seller_type',
       'transmission', 'owner', 'brand']
data = df.drop(['name', 'fuel', 'seller_type', 'transmission', 'owner', 'brand','model'], axis=1)
data.head()
def treatment_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
for i in num_cols:
    data = treatment_outliers(data, i)
from sklearn.model_selection import train_test_split
x = data.drop(columns=['selling_price'])
y = data['selling_price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.metrics import *
def eval_model(model,mname):
    model.fit(x_train,y_train)
    ypred = model.predict(x_test)
    train_r2 = model.score(x_train,y_train)
    test_r2 = model.score(x_test,y_test)
    mae = mean_absolute_error(y_test,ypred)
    mse = mean_squared_error(y_test,ypred)
    rmse = np.sqrt(mse)
    res = pd.DataFrame({'Train_R2':train_r2,'Test_R2':test_r2,'MAE':mae,
                       'MSE':mse,'RMSE':rmse},index=[mname])
    return res,ypred


from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from xgboost import XGBRegressor


# In[64]:


lr = LinearRegression()
lr_res,ypred_lr = eval_model(lr,'LinReg')
lr_res


# In[65]:


dt = DecisionTreeRegressor(max_depth=6,min_samples_split=8)
dt_res,ypred_dt = eval_model(dt,'DT_Reg')
dt_res


# In[66]:


knn = KNeighborsRegressor(n_neighbors=11)
knn_res,ypred_knn = eval_model(knn,'KNN_Reg')
knn_res


# In[67]:


rf = RandomForestRegressor(n_estimators=80,max_depth=6,min_samples_split=8)
rf_res,ypred_rf = eval_model(rf,'RF_Reg')
rf_res


# In[68]:


rg = Ridge()
rg_res,y_pred_rg = eval_model(rg,'Ridge_reg')
rg_res


# In[69]:


ls = Lasso()
ls_res,y_pred_rg = eval_model(ls,'Lasso_reg')
ls_res


# In[70]:


all_res  = pd.concat([lr_res,dt_res,knn_res,rf_res,rg_res,ls_res])
all_res


# In[71]:


import pickle
import joblib


# In[72]:


pickle.dump(rf,open('Best_Model_1.pkl','wb'))
pickle.dump(dt,open('Best_model_2.pkl','wb'))


# In[73]:


load_model=joblib.load('Best_Model_1.pkl')


# In[74]:


random_indices = np.random.choice(data.index, size= 20, replace=False)
sample_data_20 = data.loc[random_indices]
sample_data_20


# In[75]:


sample_data=sample_data_20.drop('selling_price', axis=1)


# In[76]:


Sample_pred = load_model.predict(sample_data)
Sample_pred


# In[77]:


Prediction_sample = pd.DataFrame(Sample_pred)
Prediction_sample


# In[78]:


print(sample_data_20[['selling_price', 'year']])


# In[79]:


sample_data_20['pred_selling_price']=Prediction_sample.values
sample_data_20


# In[80]:


com = (sample_data_20[['selling_price', 'pred_selling_price']])
com.to_csv('sample_prediction.csv')
com

import sklearn
print(sklearn. __version__)
