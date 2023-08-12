#!/usr/bin/env python
# coding: utf-8

# EVALUATION TECHNIQUES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# IMPORTING DATASET

# In[2]:


dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values


# In[3]:


dataset.head()


# In[4]:


print(X)


# In[5]:


print(y)


# In[6]:


from  sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])


# In[7]:


print(X)     


# In[8]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))


# In[9]:


print(X)


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# # LOGISTIC REGRESSION

# In[11]:


from sklearn.linear_model import LogisticRegression 
from sklearn import metrics


# In[12]:


reg=LogisticRegression()


# In[13]:


reg.fit(X_train,y_train)


# In[14]:


reg.score(X_train,y_train)


# In[15]:


#Create a predication set:
predications=reg.predict(X_test)


# Test the accuracy of logistic regression 

# In[16]:


#print confusion matrix
df_confusion_matrix=pd.DataFrame(metrics.confusion_matrix(y_test,predications),index=['Churn','Not Churn'],columns=['Churn','Not Churn'])
df_confusion_matrix


# In[17]:


# Print a classification report
print(metrics.classification_report(y_test,predications))


# In[18]:


# Print the overall accuracy
print(metrics.accuracy_score(y_test,predications))


# In[19]:


y_pred_proba=reg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# # Train a Naive Bayes Classifier 

# In[20]:


from sklearn.naive_bayes import MultinomialNB
nb_model=MultinomialNB()
nb_model.fit(X_train,y_train)


# In[21]:


predications=nb_model.predict(X_test)


# In[22]:


nb_model.score(X_train,y_train)


# Test the accuracy of naive bayes model

# In[23]:



df_confusion_matrix=pd.DataFrame(metrics.confusion_matrix(y_test,predications),index=['Churn','Not Churn'],columns=['Churn','Not Churn'])
df_confusion_matrix


# In[24]:


print(metrics.classification_report(y_test,predications))


# In[25]:


print(metrics.accuracy_score(y_test,predications))


# In[26]:


y_pred_proba=nb_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[27]:


print("Thank you")

