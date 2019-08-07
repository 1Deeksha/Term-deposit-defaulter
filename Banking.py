#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

df = pd.read_csv("bank.csv")

df.head(5)


# In[4]:


df.isnull().sum()


# In[6]:


from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
df['y'] = number.fit_transform(df['y'].astype('str'))
df['marital'] = number.fit_transform(df['marital'].astype('str'))
df['job'] = number.fit_transform(df['job'].astype('str'))
df['education'] = number.fit_transform(df['education'].astype('str'))
df['default'] = number.fit_transform(df['default'].astype('str'))
df['housing'] = number.fit_transform(df['housing'].astype('str'))
df['loan'] = number.fit_transform(df['loan'].astype('str'))
df['contact'] = number.fit_transform(df['contact'].astype('str'))
df['month'] = number.fit_transform(df['month'].astype('str'))
df['poutcome'] = number.fit_transform(df['poutcome'].astype('str'))

df.head(5)


# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop('y', axis=1)
y = df.y
X_train, X_test, y_train, y_test = train_test_split(df,y, test_size = 0.30)


# In[9]:


X_train.shape


# In[10]:


X_test.shape


# In[11]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(5, activation = 'relu', kernel_initializer='random_normal', input_dim=17))
#Second Hidden Layer
classifier.add(Dense(5, activation = 'relu', kernel_initializer='random_normal', input_dim=17))

#Output Layer
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal'))


# In[ ]:


#compiling the neural network
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])


# In[25]:


#fiiting train data to model
classifier.fit(X_train, y_train, batch_size=20, epochs=20)


# In[26]:


#check loss & metrics values

eval_model = classifier.evaluate(X_train, y_train)
eval_model


# In[27]:


#predicting output
#If the prediction is greater than 0.5 then the output is 1 else the output is 0

y_pred = classifier.predict(X_test)
y_pred


# In[30]:


y_pred = (y_pred > 0.5)
y_pred


# In[31]:


#checking the accuracy of test dataset
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:





# In[ ]:




