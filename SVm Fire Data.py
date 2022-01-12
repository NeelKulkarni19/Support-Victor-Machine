#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


fire= pd.read_csv('F:/Dataset/forestfires.csv')


# In[4]:


from sklearn import preprocessing


# In[5]:


label_encoder = preprocessing.LabelEncoder()


# In[6]:


fire["month"] = label_encoder.fit_transform(fire["month"])


# In[7]:


fire["day"] = label_encoder.fit_transform(fire["day"])


# In[8]:


fire["size_category"] = label_encoder.fit_transform(fire["size_category"])


# In[9]:


fire


# In[10]:


X=fire.iloc[:,:11]


# In[11]:


X


# In[12]:


y=fire["size_category"]


# In[13]:


y


# In[14]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[20]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[21]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10],'C':[15,14,13,12] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[22]:


gsv.best_params_ , gsv.best_score_


# In[23]:


clf = SVC(C= 15, gamma = 50)


# In[24]:


clf.fit(X_train , y_train)


# In[25]:


y_pred = clf.predict(X_test)


# In[26]:


acc = accuracy_score(y_test, y_pred) * 100


# In[27]:


print("Accuracy =", acc)


# In[28]:


confusion_matrix(y_test, y_pred)


# In[29]:


clf1 = SVC(C= 15, gamma = 50)


# In[30]:


clf1.fit(X , y)


# In[32]:


y_pred = clf1.predict(X)


# In[33]:


acc1 = accuracy_score(y, y_pred) * 100


# In[34]:


print("Accuracy =", acc1)


# In[35]:


confusion_matrix(y, y_pred)


# In[36]:


clf2 = SVC()


# In[37]:


param_grid = [{'kernel':['poly'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]


# In[38]:


gsv = GridSearchCV(clf,param_grid,cv=10)


# In[39]:


gsv.fit(X_train,y_train)


# In[40]:


gsv.best_params_ , gsv.best_score_


# In[41]:


clf3 = SVC()


# In[42]:


param_grid = [{'kernel':['sigmoid'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]


# In[43]:


gsv = GridSearchCV(clf,param_grid,cv=10)


# In[44]:


gsv.fit(X_train,y_train)


# In[45]:


gsv.best_params_ , gsv.best_score_


# In[ ]:




