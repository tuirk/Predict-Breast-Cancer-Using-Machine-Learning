#!/usr/bin/env python
# coding: utf-8

# # Predict Breast Cancer Using Machine Learning

# ## Data Set

# Breast Cancer Wisconsin (Diagnostic) Data Set : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data 

# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
# n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
# This database is also available through the UW CS ftp server:
# ftp ftp.cs.wisc.edu
# cd math-prog/cpo-dataset/machine-learn/WDBC/
# 
# Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 
# Attribute Information:
# 
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# 
# b) texture (standard deviation of gray-scale values)
# 
# c) perimeter
# 
# d) area
# 
# e) smoothness (local variation in radius lengths)
# 
# f) compactness (perimeter^2 / area - 1.0)
# 
# g) concavity (severity of concave portions of the contour)
# 
# h) concave points (number of concave portions of the contour)
# 
# i) symmetry
# 
# j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features. For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# ## EDA

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('breastcancerdata.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data = data.dropna(axis = 1)


# In[8]:


data.isnull().sum()


# In[9]:


data.dtypes


# In[10]:


data['diagnosis'].value_counts()


# In[11]:


sns.countplot(data['diagnosis'], label ='count')


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


LabelEncoder_Y = LabelEncoder()


# In[14]:


#transform categorical to numerical
data.iloc[:,1] = LabelEncoder_Y.fit_transform(data.iloc[:,1].values)


# In[15]:


data.iloc[:,1].values #malign transformed into 1 and benign to 0


# In[16]:


sns.pairplot(data.iloc[:,1:7], hue = "diagnosis")


# In[17]:


#correlation


# In[18]:


data.iloc[:,1:11]


# In[19]:


data.iloc[:,1:11].corr()


# In[20]:


plt.figure(figsize = (10, 10))
sns.heatmap(data.iloc[:,1:11].corr() , annot=True , fmt = '.0%')


# In[21]:


#feature scaling
# Splitting data to independent and dependent data sets
# independent --> X
# dependent --> Y


# In[22]:


X = data.iloc[:,2:31].values
Y = data.iloc[:,1].values


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.20, random_state= 0) #80-20 ratio 


# In[24]:


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[25]:


X_train


# ## Three Different Models

# Logistic Regression Classifier, Decision Tree Classifier, Random Forest Classifie

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[27]:


def models(X_train, Y_train):
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
    
    tree = DecisionTreeClassifier(criterion = 'entropy',  random_state = 0)
    tree.fit(X_train, Y_train)
    
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',  random_state = 0 )
    forest.fit(X_train, Y_train)
    
    print('Accuracy of Logistic Regression', log.score(X_train, Y_train))
    print('Accuracy of Decision Tree Classifier', tree.score(X_train, Y_train))
    print('Accuracy of Random Forest Classifier', forest.score(X_train, Y_train))
    
    return log, tree, forest


# #### Accuracy: 

# In[28]:


model = models(X_train, Y_train)


# In[29]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, model[0].predict(X_test))
true_pos = cm[0][0]
true_neg = cm[1][1]
false_pos = cm[1][0]
false_neg = cm[0][1]

print(cm)
print ('Accuracy:', (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg ))


# In[30]:


#prediction 
for i in range(len(model)):
    pred = model[i].predict(X_test)
    print('Our model prediction: ')
    print(pred)
    print()

print('Actual prediction: ')
print(Y_test)


# In[31]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
    print('Model:', i )
    print(classification_report(Y_test, model[i].predict(X_test)))
    print(accuracy_score(Y_test, model[i].predict(X_test)))
    print()


# In[ ]:




