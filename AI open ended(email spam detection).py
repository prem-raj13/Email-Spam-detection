#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("email.csv")


# In[3]:


df.head()


# In[4]:


df.isna().sum()


# In[ ]:





# In[6]:


df


# In[7]:


df.duplicated().sum()


# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


df.duplicated().sum()


# In[10]:


df.head(10)


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns


# In[12]:





# In[13]:


x=df['Message']
y=df['Category']
x_train, x_test,y_train, y_test = train_test_split(df["Message"],df["Category"],test_size = 0.2,random_state=42)


# In[14]:


cv=CountVectorizer()
x_train_vectorized = cv.fit_transform(x_train)
x_test_vectorized = cv.transform(x_test)


# In[15]:


print(x_test_vectorized)


# In[16]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train_vectorized, y_train)


# In[17]:


y_pred=nb.predict(x_test_vectorized)


# In[ ]:





# In[19]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[20]:


na=accuracy_score(y_test,y_pred)
np = precision_score(y_test, y_pred, pos_label='spam')
nr = recall_score(y_test, y_pred, pos_label='spam')
nf1 = f1_score(y_test, y_pred, pos_label='spam')

print('Accuracy:', na)
print('Precision:', np)
print('Recall:', nr)
print('F1 score:', nf1)


# In[21]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


# In[22]:


from sklearn import svm


# In[23]:


sv=svm.SVC()
sv.fit(x_train_vectorized,y_train)


# In[24]:


y_pred=sv.predict(x_test_vectorized)


# In[25]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sa=accuracy_score(y_test,y_pred)
sp = precision_score(y_test, y_pred, pos_label='spam')
sr = recall_score(y_test, y_pred, pos_label='spam')
sf1 = f1_score(y_test, y_pred, pos_label='spam')

print('Accuracy:', sa)
print('Precision:', sp)
print('Recall:', sr)
print('F1 score:', sf1)


# In[26]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


# In[27]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train_vectorized,y_train)


# In[28]:


y_pred=rf.predict(x_test_vectorized)


# In[29]:


ra=accuracy_score(y_test,y_pred)
rp = precision_score(y_test, y_pred, pos_label='spam')
rr = recall_score(y_test, y_pred, pos_label='spam')
rf1 = f1_score(y_test, y_pred, pos_label='spam')

print('Accuracy:', ra)
print('Precision:', rp)
print('Recall:', rr)
print('F1 score:', rf1)


# In[30]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


# In[31]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train_vectorized, y_train)


# In[32]:


y_pred=dt.predict(x_test_vectorized)


# In[33]:


da=accuracy_score(y_test,y_pred)
dp = precision_score(y_test, y_pred, pos_label='spam')
dr = recall_score(y_test, y_pred, pos_label='spam')
df1 = f1_score(y_test, y_pred, pos_label='spam')

print('Accuracy:', da)
print('Precision:', dp)
print('Recall:', dr)
print('F1 score:', df1)


# In[34]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


# In[36]:


algorithms = ['SVM','Random Forest','Decision Tree','Naive Bays']
accuracies = [0.9728682170542635,0.9728682170542635, 0.9660852713178295,0.9815891472868217]
plt.figure(figsize=(10,9))
sns.barplot(x=algorithms, y=accuracies, palette='Blues_r')
plt.title('Accuracy of Different Classification Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




