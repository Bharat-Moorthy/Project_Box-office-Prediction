#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[19]:


X = pd.read_csv('/final_X.csv')
y = pd.read_csv('/final_y.csv')
X.columns


# In[20]:


X=X.drop(['popularity_log','popularitylog_to_meanpopularitylog_year','budgetlog to popularity_ratio'],axis=1)
X.columns


# In[21]:


from xgboost import XGBRegressor
import time 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=42,test_size=0.20)


# In[22]:


xgb = XGBRegressor()
parameters = {
               'objective':['reg:linear'],
              'eta': [0.01, 0.03, 0.05, 0.07, 0.09], 
               'max_depth': [1, 3, 5  ],
               'min_child_weight': [1, 3 , 5,7 ],
               'subsample': [0.1,0.3, 0.4,0.6,0.8],
              'colsample_bytree': [0.4,0.5, 0.7,0.9],
              'n_estimators': [20,40,50,75,100,200],
              'gamma'            : [ 0.0, 0.1, 0.3, 0.4,0.5],
                'reg_alpha':[50,60,70,80],
                'reg_lambda':[0.5,0.6,0.7,0.8]}

xgb_random = RandomizedSearchCV(xgb,parameters,cv = 5,n_jobs = 5,scoring="neg_mean_squared_error",verbose=True,random_state=42)

xgb_random.fit(X_train,y_train)

print(xgb_random.best_score_)
print(xgb_random.best_params_)


# In[23]:


xgb_random.best_estimator_


# In[24]:


start=time.time()
Hyper_xgb=XGBRegressor(colsample_bytree=0.5, eta=0.07, gamma=0.5, min_child_weight=7,
             n_estimators=200, subsample=0.6,max_depth= 5,reg_lambda=0.8,reg_alpha=70)
Hyperxgb_model=Hyper_xgb.fit(X_train , y_train)
end=time.time()
xgb_time=end-start
y_trainpred=Hyperxgb_model.predict(X_train)
y_predxgb=Hyperxgb_model.predict(X_valid)


# In[25]:


from sklearn.metrics import mean_squared_error

ms = mean_squared_error(y_predxgb, y_valid, squared=False)
ms


# In[18]:


ms_train=mean_squared_error(y_trainpred, y_train, squared=False)
ms_train


# In[28]:


import pickle
pick_insert=open('HypertuneXGB.pkl',"wb")
pickle.dump(Hyperxgb_model,pick_insert)
pick_insert.close()


# In[ ]:




