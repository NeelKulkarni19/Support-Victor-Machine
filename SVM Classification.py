#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


SalaryTrain= pd.read_csv('F:/Dataset/SalaryData_Train(1).csv')


# In[3]:


SalaryTrain.info()


# In[4]:


SalaryTest= pd.read_csv('F:/Dataset/SalaryData_Test(1).csv')


# In[5]:


SalaryTest.info()


# In[6]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[7]:


SalaryTrain["workclass"] = label_encoder.fit_transform(SalaryTrain["workclass"])


# In[8]:


SalaryTrain["education"] = label_encoder.fit_transform(SalaryTrain["education"])


# In[9]:


SalaryTrain["maritalstatus"] = label_encoder.fit_transform(SalaryTrain["maritalstatus"])


# In[10]:


SalaryTrain["occupation"] = label_encoder.fit_transform(SalaryTrain["occupation"])


# In[11]:


SalaryTrain["relationship"] = label_encoder.fit_transform(SalaryTrain["relationship"])


# In[12]:


SalaryTrain["race"] = label_encoder.fit_transform(SalaryTrain["race"])


# In[13]:


SalaryTrain["sex"] = label_encoder.fit_transform(SalaryTrain["sex"])


# In[14]:


SalaryTrain["native"] = label_encoder.fit_transform(SalaryTrain["native"])


# In[15]:


SalaryTrain["Salary"] = label_encoder.fit_transform(SalaryTrain["Salary"])


# In[16]:


SalaryTrain


# In[17]:


SalaryTest["workclass"] = label_encoder.fit_transform(SalaryTest["workclass"])


# In[18]:


SalaryTest["education"] = label_encoder.fit_transform(SalaryTest["education"])


# In[19]:


SalaryTest["maritalstatus"] = label_encoder.fit_transform(SalaryTest["maritalstatus"])


# In[20]:


SalaryTest["occupation"] = label_encoder.fit_transform(SalaryTest["occupation"])


# In[21]:


SalaryTest["relationship"] = label_encoder.fit_transform(SalaryTest["relationship"])


# In[22]:


SalaryTest["race"] = label_encoder.fit_transform(SalaryTest["race"])


# In[23]:


SalaryTest["sex"] = label_encoder.fit_transform(SalaryTest["sex"])


# In[24]:


SalaryTest["native"] = label_encoder.fit_transform(SalaryTest["native"])


# In[25]:


SalaryTest["Salary"] = label_encoder.fit_transform(SalaryTest["Salary"])


# In[26]:


SalaryTest


# In[27]:


X_train = SalaryTrain.iloc[:,:-1]


# In[28]:


X_train


# In[29]:


y_train = SalaryTrain.iloc[:,-1]


# In[30]:


y_train


# In[31]:


X_test = SalaryTest.iloc[:,:-1]


# In[32]:


X_test


# In[33]:


y_test = SalaryTest.iloc[:,-1]


# In[34]:


y_test


# In[35]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[36]:


clf = SVC()


# In[37]:


param_grid = [{'kernel':['rbf'],'gamma':[50,5,10],'C':[15,14,13,12] }]


# In[38]:


gsv = GridSearchCV(clf,param_grid,cv=10)


# In[ ]:


gsv.fit(X_train,y_train)


# In[ ]:


clf = SVC(C= 15, gamma = 5)


# In[ ]:


clf.fit(X_train , y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


acc = accuracy_score(y_test, y_pred) * 100


# In[ ]:


print("Accuracy =", acc)


# In[ ]:





# In[ ]:


confusion_matrix(y_test, y_pred)

