#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")
print('libraries have been imported')


# In[2]:


data=pd.read_csv('cleaned_hr_data')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[ ]:





# In[6]:


data.isnull().sum() # let's drop column Unnamed


# In[7]:


data.drop(['Unnamed: 0'], inplace=True, axis=1)


# In[8]:


data.drop(['Over18'], inplace=True, axis=1)


# In[9]:


data.columns


# In[10]:


# We will perform a logistic regression to predict the outcome for test data, 
# and validate the results by using the confusion matrix.


# In[11]:


# let's copy the data frame into a new one
data1=data.copy()


# In[12]:


data1.shape


# In[13]:


data1.dtypes


# In[ ]:





# In[14]:


data1.columns


# In[15]:


data1.head()


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[17]:


# Get dummy columns for 'Attrition', 'BusinessTravel', 'Department', 'EducationField','Gender','JobRole',
#'MaritalStatus'
data1=pd.get_dummies(data1, columns = ['BusinessTravel', 'Department', 'EducationField','Gender','JobRole','MaritalStatus'])


# In[18]:


data1.dtypes


# In[19]:


data1.head()


# In[20]:


#data1.Attrition = data1.Attrition.astype(float)
#data1['Travel_Rarely'].astype(float)


# We need to split our feacture columns from our target columns 'Attrition'. 
# we will use the feactures columns to predict the target columns

# In[21]:


x=data1.drop(columns='Attrition',axis=1)
y=data1['Attrition']


# Data split into training data and testing data. first we'll need to create train and test variable for x and y

# In[22]:


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, stratify=y, random_state=3)


# In[23]:


# let's check the shape our our train and test data
print(x_train.shape, x_test.shape)


# In[24]:


# Model used: Logistic Regression
Model_var= LogisticRegression()


# In[25]:


# training  Model_var with training data
Model_var.fit(x_train, y_train)


# In[26]:


# Training accuracy:
x_train_predict = Model_var.predict(x_train)
train_accuracy = accuracy_score(x_train_predict, y_train)


# In[27]:


print('Training Accuracy: ', train_accuracy)


# In[28]:


# Test accuracy:
x_test_predict = Model_var.predict(x_test)
test_accuracy = accuracy_score(x_test_predict, y_test)


# In[29]:


print('Test Accuracy: ', test_accuracy)


# In[30]:


# accuracy calculation Confusion Matrix


# In[31]:


confusion_matrix(y_true=y_test, y_pred=x_test_predict)


# In[32]:


Model_var.classes_


# In[71]:


# a new value
data1.iloc[1].to_list()


# In[34]:


y_test.where(y_test == "Yes").dropna()


# In[35]:


x_test.where(x_test['EmployeeID'] ==  36).dropna()


# In[36]:


y_new1 = Model_var.predict(pd.DataFrame(x_test.iloc[35].tolist()).T)
y_new1


# In[37]:


y_test.iloc[35]


# In[38]:


y_test.where(y_test=='Yes').dropna()
indexes= y_test.where(y_test=='Yes').dropna().index.to_list()
indexes


# In[39]:


y_test.loc[1117]


# In[40]:


y_new = Model_var.predict(x_test.loc[indexes])
pd.Series(y_new).value_counts()


# In[41]:


12/142*100 # accuracy for Yes


# In[42]:


y_test.where(y_test=='No').dropna()
no_indexes= y_test.where(y_test=='No').dropna().index.to_list()
no_indexes


# In[43]:


y_new2 = Model_var.predict(x_test.loc[no_indexes])
pd.Series(y_new2).value_counts()


# In[44]:


728/740*100 # accuracy for No


# In[45]:


confusion_matrix(y_true=y_test, y_pred=x_test_predict)


# In[65]:


cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()


# (TN=728, FP=12), (FN=130, TP=12)

# In[70]:


accuracy=((728+12)/(728+12+12+130))*100
print(accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




