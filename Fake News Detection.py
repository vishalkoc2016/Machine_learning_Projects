#!/usr/bin/env python
# coding: utf-8

# ### Fake News Detection

# In[3]:


import pandas as pd
import numpy as np
import itertools


# In[4]:


df = pd.read.csv("news.csv")


# In[28]:


import pandas as pd

df = pd.read_csv("news.csv")


# In[29]:


import numpy as np
import itertools


# In[30]:


import pandas as pd

df = pd.read_csv("news.csv")


# In[31]:


df.head()


# In[32]:


df.shape


# In[33]:


df.isnull().sum()


# In[34]:


labels = df.label


# In[35]:


labels.head()


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(df["text"], labels, test_size=0.2, random_state=20)


# In[38]:


x_train.head()


# In[39]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


# In[40]:


vector = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[41]:


tf_train = vector.fit_transform(x_train)
tf_test = vector.transform(x_test)


# In[42]:


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tf_train, y_train)


# In[43]:


from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = pac.predict(tf_test)


# In[44]:


score = accuracy_score(y_test, y_pred)


# In[46]:


print(f"Accuracy : {round(score*100,2)}%")


# In[48]:


confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])


# In[49]:


import pickle 
filename = 'finalized_model.pkl'
pickle.dump(pac, open(filename, 'wb'))


# In[ ]:




