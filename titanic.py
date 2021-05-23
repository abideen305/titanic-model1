#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')
#this is for Back End of the matplotlib for better visualization


# In[2]:


titanic_train = pd.read_csv("titanic_train.csv")
titanic_test = pd.read_csv("titanic_test.csv")
gen_df = pd.read_csv("gender_baseline.csv")


# In[3]:


titanic_train.head()


# In[4]:


titanic_test


# In[5]:


#printing their shapes

print("The shape for train is {}.\n The shape for test is {}.".format(titanic_train.shape, titanic_test.shape))


# In[6]:


titanic_train.shape


# In[7]:


gen_df


# In[8]:


titanic_train.info()


# In[9]:


titanic_train_copy = titanic_train.copy()


# In[10]:


#del titanic_train['passenger_id']


# In[11]:


titanic_train.info()


# In[12]:


df_drop = titanic_train.drop(columns=["name", "fare", "passenger_id", "body", "home.dest", "ticket"], axis = 1)


# In[13]:


df_drop.info()


# In[14]:


df_drop = df_drop.fillna(-99999)


# In[15]:


df_drop.info()


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


#setting X for our predictors and y for what we want to predict. 

# X=titanic_train.drop(['survived'],axis=1)


# In[18]:


# y = titanic_train['survived']


# In[19]:


# #splitting with train_test_split imported earlier 

# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25)


# In[20]:


df_drop.info()


# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


lb = LabelEncoder()


# In[23]:


df_drop['sex'] =lb.fit_transform(df_drop["sex"])


# In[24]:


df_drop['boat'] = df_drop['boat'].replace('D', -99999)


# In[25]:


df_drop['boat'] = df_drop['boat'].replace('B', -999999)


# In[26]:


df_drop['boat'] = df_drop['boat'].replace('C D', -999999)


# In[27]:


df_drop['boat'] = df_drop['boat'].replace('13 15 B', -999999)


# In[28]:


df_drop['boat'] = df_drop['boat'].replace('A', -999999)


# In[29]:


df_drop['boat'] = df_drop['boat'].replace('C', -999999)


# In[30]:


df_drop['boat'] = df_drop['boat'].replace('5 7', -999999)


# In[31]:


df_drop['boat'] = df_drop['boat'].replace('13 15', -999999)


# In[32]:


df_drop['boat'] = df_drop['boat'].replace('5 9', -999999)


# In[33]:


df_drop['boat'] = df_drop['boat'].replace('15 16', -999999)


# In[34]:


df_drop["boat"] = lb.fit_transform(df_drop['boat'].astype(str))


# In[35]:


df_drop['boat'].unique()


# In[36]:



#setting X for our predictors and y for what we want to predict. 

X=df_drop.drop(['survived'],axis=1)


# In[37]:


X


# In[38]:


X.info()


# In[39]:


df_drop['embarked'] = lb.fit_transform(df_drop['embarked'].astype(str))


# In[40]:


df_drop['cabin'] = lb.fit_transform(df_drop['cabin'].astype(str))


# In[41]:


df_drop.info()


# In[42]:


plt.title("Does pclass really contributed to survival rate?")
sns.countplot(x = 'survived', hue= 'pclass', data = df_drop)
plt.show()


# In[43]:


X= df_drop.drop(columns='survived')


# In[44]:


y = df_drop['survived']


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[46]:


X_train


# In[47]:


#Let's start by using Logistic Regression model since the problem is about classification
from sklearn.linear_model import LogisticRegression
#instanciating the model
lr = LogisticRegression(max_iter=100000)


# In[48]:


#fitting our model
lr.fit(X_train, y_train)


# In[49]:


lr_pred = lr.predict(X_test)


# In[50]:


lr_pred


# In[51]:


#check the accuracy of your code
from sklearn.metrics import accuracy_score


# In[52]:


acc = accuracy_score(y_test, lr_pred)


# In[53]:


acc


# - Logistic regression gives accuraacy of 0.91 which is good

# In[54]:


from sklearn.neighbors import KNeighborsClassifier


# In[55]:


knn = KNeighborsClassifier()


# In[56]:


knn.fit(X_train, y_train)


# In[57]:


knn_pred = knn.predict(X_test)


# In[58]:


from sklearn.metrics import accuracy_score


# In[59]:


knn_acc = accuracy_score(y_test, knn_pred)


# In[60]:


knn_acc


# In[61]:


from sklearn.svm import SVC


# In[62]:


svm = SVC(kernel='linear')


# In[63]:


svm.fit(X_train, y_train)


# In[64]:


svm_pred = svm.predict(X_test)


# In[65]:


acc_svm = accuracy_score(y_test, svm_pred)


# In[66]:


acc_svm


# In[67]:


from sklearn.metrics import auc


# In[68]:


from sklearn.metrics import precision_recall_curve


# In[69]:


pre_rec_cur = precision_recall_curve(y_test, svm_pred )


# In[70]:


pre_rec_cur


# In[71]:


gen_df


# In[72]:


test_id = X_test.index


# In[73]:


test_id


# In[74]:


svm_submission = pd.DataFrame({"passenger_id": test_id, "survived": svm_pred})


# In[75]:


svm_submission_csv = svm_submission.to_csv("mySvmSubmission.csv", index=False)


# In[76]:


plt.title("How many survived and how mand died?")
plt.xlabel("Survival Rate")
plt.ylabel("Value Counts")
#plt.figure(figsize=(5,12))
sns.countplot(svm_pred)
plt.show()


# In[77]:


test_id


# In[79]:


import pickle

pickle.dump(svm,open('titanic_classifier.pkl','wb'))
model=pickle.load(open('titanic_classifier.pkl','rb'))

#print(model.predict([[1,1,22,3,1,2,39,1]]))

# In[ ]:




